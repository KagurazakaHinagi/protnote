import logging
from protnote.utils.data import log_gpu_memory_usage, read_json
from protnote.utils.evaluation import (
    EvalMetrics,
    metric_collection_to_dict_float,
    save_evaluation_results,
)
from protnote.utils.losses import (
    BatchWeightedBCE,
    FocalLoss,
    RGDBCE,
    WeightedBCE,
    SupCon,
    CBLoss,
)
from torchmetrics import MetricCollection, Metric
from protnote.utils.proteinfer import normalize_confidences
import torch.distributed as dist
import numpy as np
import torch
import wandb
import os
import pickle
import shutil
import json
from collections import defaultdict
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from transformers import BatchEncoding
from protnote.utils.models import biogpt_train_last_n_layers, save_checkpoint, load_model
from protnote.utils.interpretability import (
    load_site_ground_truth_tsv,
    gradient_input_node_attribution_from_logits,
    interpretability_loss_mse,
)
from torcheval.metrics import (
    MultilabelAUPRC,
    BinaryAUPRC,
    BinaryBinnedAUPRC,
    MultilabelBinnedAUPRC,
    Mean,
    BinaryF1Score,
)
from torcheval.metrics.toolkit import sync_and_compute


def calculate_f1_micro(total_tp_per_label, total_fn_per_label, total_fp_per_label):
    tp_micro = total_tp_per_label.sum()
    fn_micro = total_fn_per_label.sum()
    fp_micro = total_fp_per_label.sum()
    precision_micro = tp_micro / (tp_micro + fp_micro + 1e-8)
    recall_micro = tp_micro / (tp_micro + fn_micro + 1e-8)
    f1_micro = (
        2 * (precision_micro * recall_micro) / (precision_micro + recall_micro + 1e-8)
    )
    return f1_micro


def calculate_f1(tp, fn, fp):
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1


def calculate_tp_fn_fp(probs, labels, threshold=0.5):
    """
    Calculate true positives, false negatives, and false positives per label.

    Args:
        probs (torch.Tensor): A tensor of probabilities with shape (num_observations, num_labels).
        labels (torch.Tensor): A tensor of true labels with shape (num_observations, num_labels).
        threshold (float): The threshold to convert probabilities to binary predictions.

    Returns:
        tp (torch.Tensor): True positives per label.
        fn (torch.Tensor): False negatives per label.
        fp (torch.Tensor): False positives per label.
    """
    # Convert probabilities to binary predictions
    preds = (probs >= threshold).float()

    # Calculate true positives, false negatives, and false positives per label
    tp = (preds * labels).sum(dim=0)
    fn = ((1 - preds) * labels).sum(dim=0)
    fp = (preds * (1 - labels)).sum(dim=0)

    return tp, fn, fp


class ProtNoteTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: str,
        rank: int,
        config: dict,
        logger: logging.Logger,
        timestamp: str,
        run_name: str,
        loss_fn: torch.nn.Module,
        use_wandb: bool = False,
        use_amlt: bool = False,
        is_master: bool = True,
        starting_epoch: int = 1,
        trial_callback=None,
    ):
        """
        Args:
            model (nn.Module): The PyTorch model to train.
            device (str): The device to use for training (e.g., 'cpu' or 'cuda').
            logger (logging.Logger): The logger to use for logging training progress.
            timestamp (str): The timestamp to use for naming log files and checkpoints.
            run_name (str): The name of the current training run.
            use_wandb (bool, optional): Whether to use Weights & Biases for logging. Defaults to False.
            bce_pos_weight (torch.Tensor, optional): The positive weight for binary cross-entropy loss. Defaults to None.
            is_master (bool, optional): Whether the current process is the master process. Defaults to True.
            starting_epoch (int, optional): The starting epoch number. Defaults to 1. Used for resuming training.
        """

        self.model = model
        self.is_master = is_master
        self.device = device
        self.rank = rank
        self.run_name = run_name
        self.logger = logger
        self.timestamp = timestamp
        self.use_wandb = use_wandb
        self.use_amlt = use_amlt
        self.loss_fn = loss_fn
        self.best_val_metric = 0.0  # WARNING: Assumes higher is better
        self.best_val_loss = float("inf")
        self.starting_epoch = starting_epoch
        self.trial_callback = trial_callback
        self.epoch = starting_epoch
        self.config = config
        self.num_epochs = config["params"]["NUM_EPOCHS"]
        self.train_sequence_encoder = config["params"]["TRAIN_SEQUENCE_ENCODER"]
        self.use_structural_encoder = config["params"].get("USE_STRUCTURAL_ENCODER", False)
        self.interpretability_method = (config["params"].get("INTERPRETABILITY_METHOD", "none") or "none").strip().lower()
        self.interpretability_weight = config["params"].get("INTERPRETABILITY_WEIGHT", 0.1)
        self.interpretability_n_steps = config["params"].get("INTERPRETABILITY_N_STEPS", 50)
        self._site_ground_truth = None
        if self.interpretability_method == "integratedgradient" and self.use_structural_encoder:
            tsv_path = config["paths"].get("INTERPRETABILITY_SITE_TSV_PATH")
            if tsv_path and os.path.isfile(tsv_path):
                self._site_ground_truth = load_site_ground_truth_tsv(tsv_path)
                self.logger.info(f"Interpretability: loaded site ground truth for {len(self._site_ground_truth)} entries from {tsv_path}")
            else:
                self.logger.warning(
                    "INTERPRETABILITY_METHOD=IntegratedGradient but INTERPRETABILITY_SITE_TSV_PATH missing or not a file; interpretability loss disabled."
                )
                self._site_ground_truth = {}
        self.label_encoder_num_trainable_layers = config["params"][
            "LABEL_ENCODER_NUM_TRAINABLE_LAYERS"
        ]
        self.train_projection_head = config["params"]["TRAIN_PROJECTION_HEAD"]
        self.normalize_probabilities = config["params"]["NORMALIZE_PROBABILITIES"]
        self.EPOCHS_PER_VALIDATION = config["params"]["EPOCHS_PER_VALIDATION"]
        self.gradient_accumulation_steps = config["params"][
            "GRADIENT_ACCUMULATION_STEPS"
        ]
        self.clip_value = config["params"]["CLIP_VALUE"]
        self.label_normalizer = read_json(config["paths"]["PARENTHOOD_LIB_PATH"])
        self.output_model_dir = config["paths"]["OUTPUT_MODEL_DIR"]
        self.lora_params = (
            {
                "rank": config["params"]["LORA_RANK"],
                "alpha": config["params"]["LORA_ALPHA"],
                "in_features": config["params"]["LABEL_EMBEDDING_DIM"],
                "out_features": config["params"]["LABEL_EMBEDDING_DIM"],
                "device": self.device,
            }
            if config["params"]["LORA"]
            else None
        )

        self._set_optimizer(
            opt_name=config["params"]["OPTIMIZER"], lr=config["params"]["LEARNING_RATE"]
        )

        self.scaler = GradScaler("cuda")
        self.base_model_path = self._get_saved_model_base_path()
        self.model_path_best_metric = self.base_model_path + f"_best_val_metric.pt"
        self.model_path_best_loss = self.base_model_path + f"_best_val_loss.pt"
        self.model_path_last_epoch = self.base_model_path + f"_last_epoch.pt"

        # self.tb = SummaryWriter(f"runs/{self.run_name}_{self.timestamp}") if self.is_master else None

    def _get_saved_model_base_path(self):
        # Save model to OUTPUT_MODEL_DIR. Create path if it doesn't exist.
        if not os.path.exists(self.output_model_dir) and self.is_master:
            os.makedirs(self.output_model_dir)

        model_name = self.run_name if self.run_name else "ProtNote"
        model_path = os.path.join(
            self.output_model_dir, f"{self.timestamp}_{model_name}"
        )
        return model_path

    def _to_device(self, *args):
        processed_args = []
        for item in args:
            if item is None:
                processed_args.append(item)
            elif isinstance(item, torch.Tensor):
                processed_args.append(item.to(self.device))
            elif isinstance(item, BatchEncoding) or isinstance(item, dict):
                processed_dict = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in item.items()
                }
                processed_args.append(processed_dict)
            elif hasattr(item, "to"):  # PyG Batch
                processed_args.append(item.to(self.device))
            else:
                processed_args.append(item)
        return processed_args

    def _get_model(self):
        if hasattr(self.model, "module"):
            return self.model.module
        return self.model

    def _set_optimizer(self, opt_name, lr):
        trainable_params = []
        trainable_params_names = []

        # Use to unfreeze last n layers. 0 means entire model frozen.
        biogpt_train_last_n_layers(
            self._get_model().label_encoder,
            self.label_encoder_num_trainable_layers,
            lora_params=self.lora_params,
        )

        for name, param in self._get_model().named_parameters():
            if name.startswith("sequence_encoder") and (
                not self.train_sequence_encoder
            ):
                param.requires_grad = False

            if (name.startswith("W_p.") or name.startswith("W_l.")) and (
                not self.train_projection_head
            ):
                param.requires_grad = False

            if name.startswith("output_layer") and (not self.train_projection_head):
                param.requires_grad = False

            if param.requires_grad:
                trainable_params.append(param)
                trainable_params_names.append(name)

        self.trainable_params_names = trainable_params_names

        if opt_name == "Adam":
            self.optimizer = torch.optim.Adam(trainable_params, lr=lr)
        elif opt_name == "AdamW":
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=lr,
                weight_decay=self.config["params"]["WEIGHT_DECAY"],
            )
        elif opt_name == "SGD":
            self.optimizer = torch.optim.SGD(
                trainable_params,
                lr=lr,
                weight_decay=self.config["params"]["WEIGHT_DECAY"],
            )
        else:
            raise ValueError("Unsupported optimizer name")

    def evaluation_step(self, batch, return_embeddings=False) -> tuple:
        """Perform a single evaluation step.

        :param batch: _description_
        :type batch: _type_
        :return: batch loss, logits and labels
        :rtype: tuple
        """
        if "graph_data" in batch:
            # Atom-level format (ESM-C + EGNN)
            graph_data = batch["graph_data"]
            sequence_ids = batch.get("sequence_ids", [])
            label_multihots = batch["label_multihots"]
            label_embeddings = batch["label_embeddings"]
            # Move graph_data tensors to device
            graph_data = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in graph_data.items()}
            label_multihots, label_embeddings = self._to_device(label_multihots, label_embeddings)
            inputs = {
                "graph_data": graph_data,
                "label_embeddings": label_embeddings,
            }
        elif self.use_structural_encoder:
            structure_batch = batch["structure_batch"]
            sequence_ids = batch.get("sequence_ids", [])
            label_multihots = batch["label_multihots"]
            label_embeddings = batch["label_embeddings"]
            structure_batch, label_multihots, label_embeddings = self._to_device(
                structure_batch, label_multihots, label_embeddings
            )
            inputs = {
                "structure_batch": structure_batch,
                "label_embeddings": label_embeddings,
            }
        else:
            (
                sequence_onehots,
                sequence_lengths,
                sequence_ids,
                label_multihots,
                label_embeddings,
            ) = (
                batch["sequence_onehots"],
                batch["sequence_lengths"],
                batch["sequence_ids"],
                batch["label_multihots"],
                batch["label_embeddings"],
            )
            (
                sequence_onehots,
                sequence_lengths,
                label_multihots,
                label_embeddings,
            ) = self._to_device(
                sequence_onehots, sequence_lengths, label_multihots, label_embeddings
            )
            inputs = {
                "sequence_onehots": sequence_onehots,
                "sequence_lengths": sequence_lengths,
                "label_embeddings": label_embeddings,
            }

        # Forward pass
        with autocast(device_type="cuda"):
            logits, embeddings = self.model(**inputs, save_embeddings=return_embeddings)
            loss = self.loss_fn(logits, label_multihots.float())

        return loss, logits, label_multihots, sequence_ids, embeddings

    def validate(
        self,
        val_loader: torch.utils.data.DataLoader,
        eval_metrics: MetricCollection,
        val_optimization_metric_name: str,
        only_represented_labels: bool,
    ):
        self.logger.info("Running validation...")

        prefix = "validation"

        val_metrics = self.evaluate(
            data_loader=val_loader,
            eval_metrics=eval_metrics,
            data_loader_name=prefix,
            only_represented_labels=only_represented_labels,
        )
        val_optimization_metric_name = f"{prefix}_{val_optimization_metric_name}"

        self.logger.info(
            "+-------------------------------- Validation Results --------------------------------+"
        )
        # Print memory consumption
        if self.is_master:
            log_gpu_memory_usage(self.logger, 0)
        self.logger.info(f"Validation metrics:\n{json.dumps(val_metrics, indent=4)}")

        if self.use_wandb and self.is_master:
            try:
                if self.use_wandb and self.is_master:
                    wandb.log(val_metrics, step=self.training_step)

            except Exception as e:
                self.logger.warning(f"Failed to log validation metrics to wandb: {e}")

        # Save the model if it has the best validation **metric** so far (only on master node)
        if (
            self.is_master
            and val_metrics[val_optimization_metric_name] > self.best_val_metric
        ):
            self.logger.info(
                f"New best {val_optimization_metric_name}: {val_metrics[val_optimization_metric_name]}. Saving model..."
            )
            self.best_val_metric = val_metrics[val_optimization_metric_name]

            save_checkpoint(
                model=self._get_model(),
                optimizer=self.optimizer,
                epoch=self.epoch,
                best_val_metric=self.best_val_metric,
                best_val_loss=self.best_val_loss,
                model_path=self.model_path_best_metric,
            )
            self.logger.info(f"Saved model to {self.model_path_best_metric}")

            if self.use_wandb:
                wandb.save(
                    f"{self.timestamp}_best_{val_optimization_metric_name}_ProtNote.pt"
                )

        # Save the model if it has the best validation **loss** so far (only on master node)
        if self.is_master and val_metrics[f"{prefix}_loss"] < self.best_val_loss:
            self.logger.info(
                f"New best loss: {val_metrics[f'{prefix}_loss']}. Saving model..."
            )
            self.best_val_loss = val_metrics[f"{prefix}_loss"]

            save_checkpoint(
                model=self._get_model(),
                optimizer=self.optimizer,
                epoch=self.epoch,
                best_val_metric=self.best_val_loss,
                best_val_loss=self.best_val_loss,
                model_path=self.model_path_best_loss,
            )
            self.logger.info(f"Saved model to {self.model_path_best_loss}")

            if self.use_wandb:
                wandb.save(f"{self.timestamp}_best_loss_ProtNote.pt")

        self.logger.info(
            "+------------------------------------------------------------------------------------+"
        )

        return val_metrics

    def find_optimal_threshold(
        self, data_loader: torch.utils.data.DataLoader, optimization_metric_name: str
    ) -> tuple[float, float]:
        """Find the optimal threshold for the given data loader.

        :param data_loader: _description_
        :type data_loader: torch.utils.data.DataLoader
        :param average: _description_
        :type average: Literal[&#39;micro&#39;, &#39;macro&#39;, &#39;weighted&#39;]
        :param optimization_metric_name: _description_
        :type optimization_metric_name: str
        :return: _description_
        :rtype: tuple[float, float]
        """

        self.logger.info("Finding optimal threshold...")
        self.model.eval()

        best_th = 0.0
        best_score = 0.0
        all_probabilities = []
        all_label_multihots = []

        all_probabilities = []
        all_label_multihots = []

        with torch.no_grad():
            for batch in data_loader:
                _, logits, label_multihots, _, embeddings = self.evaluation_step(
                    batch=batch
                )

                # Apply sigmoid to get the probabilities for multi-label classification
                probabilities = torch.sigmoid(logits)

                if self.normalize_probabilities:
                    probabilities = self._normalize_probabilities(probabilities)
                all_probabilities.append(probabilities.cpu())
                all_label_multihots.append(label_multihots.cpu())

            all_probabilities = torch.cat(all_probabilities)
            all_label_multihots = torch.cat(all_label_multihots)

        for th in np.arange(0.1, 1, 0.01):
            optimization_metric = EvalMetrics(device="cpu").get_metric_by_name(
                name=optimization_metric_name,
                threshold=th,
                num_labels=all_label_multihots.shape[-1],
            )

            optimization_metric(all_probabilities, all_label_multihots)
            score = optimization_metric.compute().item()
            if score > best_score:
                best_score = score
                best_th = float(th)
            self.logger.info("TH: {:.3f}, F1: {:.3f}".format(th, score))

        best_score = best_score
        self.logger.info(
            f"Best validation score: {best_score}, Best val threshold: {best_th}"
        )
        self.model.train()
        return best_th, best_score

    def _normalize_probabilities(self, probabilities):
        # TODO: Using original normalize_confidences implemented with numpy,
        # but this is slow. Should be able to do this with torch tensors.
        """
        return torch.tensor(
                    normalize_confidences(
                        predictions=probabilities.detach().cpu().numpy(),
                        label_vocab=self.vocabularies["GO_label_vocab"],
                        applicable_label_dict=self.label_normalizer,
                    ),
                    device=self.device,
                )
        """

    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader,
        eval_metrics: MetricCollection = None,
        save_results: bool = False,
        data_loader_name=None,
        only_represented_labels: bool = False,
        return_embeddings: bool = False,
    ) -> tuple[dict, dict]:
        """Evaluate the model on the given data loader.
        :param data_loader: pytorch data loader
        :type data_loader: torch.utils.data.DataLoader
        :param eval_metrics: an eval metrics class to calculate metrics like F1 score, defaults to None
        :type eval_metrics: EvalMetrics, optional
        :return: dictionary with evaluation metrics. Always return avg_loss and if eval_metrics is not None, it will return the metrics from eval_metrics.compute()
        :rtype: dict
        """
        self.model.eval()
        test_results = defaultdict(list)

        if only_represented_labels:
            num_labels = sum(data_loader.dataset.represented_vocabulary_mask)
        else:
            num_labels = len(data_loader.dataset.label_vocabulary)

        if eval_metrics is not None:
            eval_metrics.reset()

        if self.config["params"]["ESTIMATE_MAP"] == False:
            mAP_micro = BinaryAUPRC(device="cpu")
            mAP_macro = MultilabelAUPRC(device="cpu", num_labels=num_labels)

        elif self.config["params"]["ESTIMATE_MAP"] == True:
            mAP_micro = BinaryBinnedAUPRC(device="cpu", threshold=50)
            mAP_macro = MultilabelBinnedAUPRC(
                device="cpu", num_labels=num_labels, threshold=50
            )

        elif self.config["params"]["ESTIMATE_MAP"] is None:
            self.logger.info("Not computing mAP metrics")
            mAP_macro = mAP_micro = None

        avg_loss = Mean(device=self.device)
        total_tp_per_label = torch.zeros(num_labels, device=self.device)
        total_fn_per_label = torch.zeros(num_labels, device=self.device)
        total_fp_per_label = torch.zeros(num_labels, device=self.device)
        all_embeddings = {
            "output_layer_embeddings": [],
            "joint_embeddings": [],
            "labels": [],
            "sequence_ids": [],
        }
        embeddings_num_batches = 100  # The number of embedding batches to export at a time if return_embedding = True
        embeddings_export_dir = os.path.join(
            self.config["paths"]["RESULTS_DIR"],
            f"{data_loader_name}_embeddings_{self.run_name}",
        )
        if return_embeddings:
            if os.path.exists(embeddings_export_dir):
                shutil.rmtree(embeddings_export_dir)
            os.mkdir(embeddings_export_dir)

        batch_idx = -1
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                loss, logits, labels, sequence_ids, embeddings = self.evaluation_step(
                    batch=batch, return_embeddings=return_embeddings
                )
                if only_represented_labels:
                    logits = logits[:, data_loader.dataset.represented_vocabulary_mask]
                    labels = labels[:, data_loader.dataset.represented_vocabulary_mask]

                if eval_metrics is not None:
                    # Apply sigmoid to get the probabilities for multi-label classification
                    probabilities = torch.sigmoid(logits)

                    if self.normalize_probabilities:
                        probabilities = self._normalize_probabilities(probabilities)

                    # Update eval metrics
                    eval_metrics(probabilities, labels)
                    if self.config["params"]["DECISION_TH"] is not None:
                        tp, fn, fp = calculate_tp_fn_fp(
                            probs=probabilities,
                            labels=labels,
                            threshold=self.config["params"]["DECISION_TH"],
                        )

                        total_tp_per_label += tp
                        total_fn_per_label += fn
                        total_fp_per_label += fp

                    if (mAP_macro is not None) and (mAP_micro is not None):
                        mAP_micro.update(
                            probabilities.cpu().flatten(), labels.cpu().flatten()
                        )
                        mAP_macro.update(probabilities.cpu(), labels.cpu())

                # No need to save results everytime. Only need it for final evaluation.
                if save_results:
                    test_results["sequence_ids"].append(sequence_ids)
                    test_results["logits"].append(logits.cpu())
                    test_results["labels"].append(labels.cpu())

                if return_embeddings:
                    all_embeddings["joint_embeddings"].append(
                        embeddings["joint_embeddings"]
                    )
                    all_embeddings["output_layer_embeddings"].append(
                        embeddings["output_layer_embeddings"]
                    )
                    all_embeddings["labels"].append(labels.cpu())
                    all_embeddings["sequence_ids"].append(sequence_ids)

                    # Export every 100 batches

                    if (batch_idx + 1) % embeddings_num_batches == 0:
                        for key, embedding_list in all_embeddings.items():
                            if key == "sequence_ids":
                                all_embeddings[key] = [
                                    j
                                    for i in all_embeddings["sequence_ids"]
                                    for j in i
                                ]
                            else:
                                all_embeddings[key] = torch.cat(
                                    embedding_list
                                ).numpy()

                        torch.save(
                            all_embeddings,
                            os.path.join(
                                embeddings_export_dir,
                                f"batches_{batch_idx-embeddings_num_batches+1}_{batch_idx}.pt",
                            ),
                            pickle_protocol=pickle.HIGHEST_PROTOCOL,
                        )

                        # Clean buffer
                        all_embeddings = {k: [] for k in all_embeddings.keys()}

                # Print progress every 25%
                progress_chunk = 20
                if (
                    batch_idx
                    % (max(len(data_loader), progress_chunk) // progress_chunk)
                    == 0
                ):
                    self.logger.info(
                        f"[Evaluation] Epoch {self.epoch}: Processed {batch_idx} out of {len(data_loader)} batches ({batch_idx / len(data_loader) * 100:.2f}%)."
                    )

                # Update loss
                avg_loss.update(loss)

            if return_embeddings and any(len(v) > 0 for v in all_embeddings.values()):
                for key, embedding_list in all_embeddings.items():
                    if key == "sequence_ids":
                        all_embeddings[key] = [j for i in embedding_list for j in i]
                    else:
                        all_embeddings[key] = torch.cat(embedding_list).numpy()
                torch.save(
                    all_embeddings,
                    os.path.join(
                        embeddings_export_dir,
                        f"batches_{batch_idx - (batch_idx % embeddings_num_batches)}_{batch_idx}.pt",
                    ),
                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                )

            if save_results:
                for key in test_results.keys():
                    if key == "sequence_ids":
                        test_results[key] = np.array(
                            [j for i in test_results["sequence_ids"] for j in i]
                        )
                    else:
                        test_results[key] = torch.cat(test_results[key]).numpy()

                self.logger.info("Saving validation results...")
                if self.is_master:
                    save_evaluation_results(
                        results=test_results,
                        label_vocabulary=[
                            i
                            for i, mask in zip(
                                data_loader.dataset.label_vocabulary,
                                data_loader.dataset.represented_vocabulary_mask,
                            )
                            if mask == True
                        ] if only_represented_labels else data_loader.dataset.label_vocabulary,
                        run_name=self.run_name,
                        output_dir=self.config["paths"]["RESULTS_DIR"],
                        data_split_name=data_loader_name,
                        save_as_h5=True
                    )

            # Aggregate the TP, FN, FP across all GPUs
            if dist.is_initialized():
                dist.reduce(total_tp_per_label, dst=0, op=dist.ReduceOp.SUM)
                dist.reduce(total_fn_per_label, dst=0, op=dist.ReduceOp.SUM)
                dist.reduce(total_fp_per_label, dst=0, op=dist.ReduceOp.SUM)

            global_f1_scores_per_label = calculate_f1(
                tp=total_tp_per_label, fn=total_fn_per_label, fp=total_fp_per_label
            )
            global_f1_macro = global_f1_scores_per_label.mean()
            global_f1_micro = calculate_f1_micro(
                total_tp_per_label=total_tp_per_label,
                total_fn_per_label=total_fn_per_label,
                total_fp_per_label=total_fp_per_label,
            )

            final_metrics = eval_metrics.compute() if eval_metrics is not None else {}

            global_mAP_micro = sync_and_compute(mAP_micro) if mAP_micro is not None else None
            global_avg_loss = sync_and_compute(avg_loss)
            global_mAP_macro = sync_and_compute(mAP_macro) if mAP_macro is not None else None

            final_metrics.update(
                {
                    "loss": global_avg_loss,
                    "map_micro": global_mAP_micro,
                    "map_macro": global_mAP_macro,
                    "f1_macro": global_f1_macro,
                    "f1_micro": global_f1_micro,
                }
            )

            final_metrics = metric_collection_to_dict_float(
                final_metrics, prefix=data_loader_name
            )

        self.model.train()

        return final_metrics

    def train_one_epoch(
        self, train_loader: torch.utils.data.DataLoader, eval_metrics: MetricCollection
    ):
        avg_loss = Mean(device=self.device)
        num_labels = len(train_loader.dataset.label_vocabulary)
        total_tp_per_label = torch.zeros(num_labels, device=self.device)
        total_fn_per_label = torch.zeros(num_labels, device=self.device)
        total_fp_per_label = torch.zeros(num_labels, device=self.device)
        eval_metrics.reset()

        ####### TRAINING LOOP #######
        for batch_idx, batch in enumerate(train_loader):
            self.training_step += 1

            # Unpack the training batch (atom-level, structural, or sequence mode)
            if "graph_data" in batch:
                # Atom-level format (ESM-C + EGNN)
                graph_data = batch["graph_data"]
                sequence_ids = batch.get("sequence_ids", [])
                label_multihots = batch["label_multihots"]
                label_embeddings = batch["label_embeddings"]
                label_token_counts = batch["label_token_counts"]
                # Move graph_data tensors to device
                graph_data = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in graph_data.items()}
                label_multihots, label_embeddings, label_token_counts = self._to_device(
                    label_multihots, label_embeddings, label_token_counts
                )
                inputs = {
                    "graph_data": graph_data,
                    "label_embeddings": label_embeddings,
                    "label_token_counts": label_token_counts,
                }
            elif self.use_structural_encoder:
                structure_batch = batch["structure_batch"]
                sequence_ids = batch.get("sequence_ids", [])
                label_multihots = batch["label_multihots"]
                label_embeddings = batch["label_embeddings"]
                label_token_counts = batch["label_token_counts"]
                structure_batch, label_multihots, label_embeddings, label_token_counts = self._to_device(
                    structure_batch, label_multihots, label_embeddings, label_token_counts
                )
                if self._site_ground_truth is not None and getattr(structure_batch, "plm", None) is not None:
                    structure_batch.plm.requires_grad_(True)
                inputs = {
                    "structure_batch": structure_batch,
                    "label_embeddings": label_embeddings,
                    "label_token_counts": label_token_counts,
                }
            else:
                (
                    sequence_onehots,
                    sequence_lengths,
                    label_multihots,
                    label_embeddings,
                    label_token_counts,
                ) = (
                    batch["sequence_onehots"],
                    batch["sequence_lengths"],
                    batch["label_multihots"],
                    batch["label_embeddings"],
                    batch["label_token_counts"],
                )
                (
                    sequence_onehots,
                    sequence_lengths,
                    label_multihots,
                    label_embeddings,
                    label_token_counts,
                ) = self._to_device(
                    sequence_onehots,
                    sequence_lengths,
                    label_multihots,
                    label_embeddings,
                    label_token_counts,
                )
                inputs = {
                    "sequence_onehots": sequence_onehots,
                    "sequence_lengths": sequence_lengths,
                    "label_embeddings": label_embeddings,
                    "label_token_counts": label_token_counts,
                }

            # Forward pass
            with autocast(device_type="cuda"):
                logits, _ = self.model(**inputs)

                # Compute loss, normalized by the number of gradient accumulation steps
                loss = (
                    self.loss_fn(logits, label_multihots.float())
                    / self.gradient_accumulation_steps
                )

                # Interpretability loss (IntegratedGradient): MSE between gradient-input attribution and site ground truth
                if (
                    self.use_structural_encoder
                    and self._site_ground_truth is not None
                    and self.interpretability_method == "integratedgradient"
                    and getattr(structure_batch, "plm", None) is not None
                    and structure_batch.plm.requires_grad
                ):
                    node_attr = gradient_input_node_attribution_from_logits(
                        logits, structure_batch, baseline_plm=None, device=self.device
                    )
                    interp_loss, _ = interpretability_loss_mse(
                        node_attr,
                        structure_batch.batch,
                        sequence_ids,
                        self._site_ground_truth,
                        self.device,
                        logits_per_sample=logits.sum(dim=1),
                    )
                    loss = loss + self.interpretability_weight * interp_loss

            # Backward pass with mixed precision
            self.scaler.scale(loss).backward()

            # Gradient accumulation every GRADIENT_ACCUMULATION_STEPS
            if (self.training_step % self.gradient_accumulation_steps == 0) or (
                batch_idx + 1 == len(train_loader)
            ):
                # Unscales the gradients of optimizer's assigned params in-place
                self.scaler.unscale_(self.optimizer)

                # Apply gradient clipping
                if self.clip_value is not None:
                    clip_grad_norm_(
                        self._get_model().parameters(), max_norm=self.clip_value
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            avg_loss.update(loss.detach() * self.gradient_accumulation_steps)

            eval_metrics(
                torch.sigmoid(logits.detach()), label_multihots.detach()
            )  # detaching labels is not "necessary" because they don't retain the graph
            if self.config["params"]["DECISION_TH"] is not None:
                tp, fn, fp = calculate_tp_fn_fp(
                    probs=torch.sigmoid(logits.detach()),
                    labels=label_multihots.detach(),
                    threshold=self.config["params"]["DECISION_TH"],
                )

                total_tp_per_label += tp
                total_fn_per_label += fn
                total_fp_per_label += fp

            if self.use_wandb and self.is_master:
                wandb.log(
                    {"per_batch_train_loss": loss.item() * self.gradient_accumulation_steps},
                    step=self.training_step,
                )

            # Print memory consumption after first batch (to get the max memory consumption during training)
            if batch_idx == 1 and self.is_master:
                self.logger.info(
                    "+----------------- Train GPU Memory Usage -----------------+"
                )
                log_gpu_memory_usage(self.logger, 0)
                self.logger.info(
                    "+----------------------------------------------------------+"
                )

            # Print progress every 10%
            if batch_idx % max(len(train_loader) // 10, 1) == 0:
                self.logger.info(
                    f"[Train] Epoch {self.epoch}: Processed {batch_idx} out of {len(train_loader)} batches ({batch_idx / len(train_loader) * 100:.2f}%)."
                )

        # Aggregate the TP, FN, FP across all GPUs
        if dist.is_initialized():
            dist.reduce(total_tp_per_label, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(total_fn_per_label, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(total_fp_per_label, dst=0, op=dist.ReduceOp.SUM)

        global_f1_scores_per_label = calculate_f1(
            tp=total_tp_per_label, fn=total_fn_per_label, fp=total_fp_per_label
        )

        global_f1_macro = global_f1_scores_per_label.mean()
        global_f1_micro = calculate_f1_micro(
            total_tp_per_label=total_tp_per_label,
            total_fn_per_label=total_fn_per_label,
            total_fp_per_label=total_fp_per_label,
        )
        global_avg_train_loss = sync_and_compute(avg_loss)

        train_metrics = eval_metrics.compute() if eval_metrics is not None else {}
        train_metrics.update(
            {
                "loss": global_avg_train_loss,
                "f1_macro": global_f1_macro,
                "f1_micro": global_f1_micro,
            }
        )

        train_metrics = metric_collection_to_dict_float(train_metrics, prefix="train")

        if self.use_wandb and self.is_master:
            wandb.log(train_metrics, step=self.training_step)

        return train_metrics

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        train_eval_metrics: MetricCollection,
        val_eval_metrics: MetricCollection,
        val_optimization_metric_name: str,
        only_represented_labels: bool,
    ):
        """Train model
        :param train_loader: training set dataloader
        :type train_loader: torch.utils.data.DataLoader
        :param val_loader: validation set dataloader
        :type val_loader: torch.utils.data.DataLoader
        :param val_optimization_metric_name: metric name  used to save checkpoints based on validation performance
        :type val_optimization_metric_name: str
        """
        self.model.train()

        # Watch the model
        if self.use_wandb:
            wandb.watch(self.model)

        # Compute total number of training steps (account for resumed training)
        self.training_step = (self.starting_epoch - 1) * len(train_loader)
        num_training_steps = len(train_loader) * self.num_epochs

        self.logger.info(f"{'='*100}")
        self.logger.info(
            f"Starting training. Total number of training steps: {num_training_steps}"
        )
        self.logger.info(f"{'='*100}")

        for epoch in range(self.starting_epoch, self.starting_epoch + self.num_epochs):
            self.logger.info(
                f"Starting epoch {epoch}/{self.starting_epoch + self.num_epochs - 1}..."
            )
            self.epoch = epoch

            # Set distributed loader epoch to shuffle data
            if hasattr(train_loader.batch_sampler, "set_epoch"):
                train_loader.batch_sampler.set_epoch(epoch)
            elif hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            train_metrics = self.train_one_epoch(
                train_loader=train_loader, eval_metrics=train_eval_metrics
            )

            if epoch % self.EPOCHS_PER_VALIDATION == 0:
                ####### VALIDATION LOOP #######
                torch.cuda.empty_cache()

                # Run validation
                self.validate(
                    val_loader=val_loader,
                    eval_metrics=val_eval_metrics,
                    val_optimization_metric_name=val_optimization_metric_name,
                    only_represented_labels=only_represented_labels,
                )

                # Optuna HPO: report metric and check for pruning.
                if self.trial_callback is not None:
                    if dist.is_initialized() and dist.get_world_size() > 1:
                        # Multi-GPU: broadcast stop signal to all ranks
                        should_stop = torch.tensor([0], device=self.device)
                        if self.is_master:
                            try:
                                self.trial_callback(epoch, self.best_val_metric)
                            except Exception:
                                should_stop.fill_(1)
                        dist.broadcast(should_stop, src=0)
                        if should_stop.item():
                            from optuna.exceptions import TrialPruned
                            raise TrialPruned()
                    elif self.is_master:
                        # Single-GPU: call directly
                        self.trial_callback(epoch, self.best_val_metric)

                self.logger.info(
                    f"Epoch {epoch}/{self.starting_epoch + self.num_epochs - 1}, Batch {self.training_step}, Training Loss: {train_metrics['train_loss']}"
                )

            # Save model every 10 epochs and the last epoch
            if self.is_master:
                if epoch == self.starting_epoch + self.num_epochs - 1:
                    self.logger.info("Saving model from last epoch...")
                    save_checkpoint(
                        model=self._get_model(),
                        optimizer=self.optimizer,
                        epoch=self.epoch,
                        best_val_metric=self.best_val_metric,
                        best_val_loss=self.best_val_loss,
                        model_path=self.model_path_last_epoch,
                    )
                    self.logger.info(f"Saved model to {self.model_path_last_epoch}")

                    if self.use_wandb:
                        wandb.save(f"{self.timestamp}_last_epoch_ProtNote.pt")

                if epoch % 10 == 0:
                    self.logger.info(f"Saving checkpoint from epoch {epoch}...")
                    epoch_model_path = self.base_model_path + f"_epoch_{epoch}.pt"
                    save_checkpoint(
                        model=self._get_model(),
                        optimizer=self.optimizer,
                        epoch=self.epoch,
                        best_val_metric=self.best_val_metric,
                        best_val_loss=self.best_val_loss,
                        model_path=epoch_model_path,
                    )
                    self.logger.info(f"Saved model to {epoch_model_path}")

                    if self.use_wandb:
                        wandb.save(os.path.basename(epoch_model_path))

        if self.is_master:
            if os.path.exists(self.model_path_best_metric):
                self.logger.info(
                    f"Restoring model to best validation {val_optimization_metric_name}..."
                )
                load_model(
                    trainer=self,
                    rank=self.rank,
                    checkpoint_path=self.model_path_best_metric,
                )
            else:
                self.logger.warning(
                    f"Best validation checkpoint not found at {self.model_path_best_metric}. "
                    f"Keeping last epoch weights. This usually means validation metrics never improved from 0."
                )

            # Broadcast model state to other processes
            if dist.is_initialized():
                for param in self._get_model().parameters():
                    # src=0 means source is the master process
                    torch.distributed.broadcast(param.data, src=0)
        else:
            # For non-master processes, just receive the broadcasted data
            if dist.is_initialized():
                for param in self._get_model().parameters():
                    torch.distributed.broadcast(param.data, src=0)

        # self.tb.close()
