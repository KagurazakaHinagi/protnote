import json
import os
from pathlib import Path
import socket

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModel, AutoTokenizer

import wandb
from protnote.data.datasets import (
    ProteinDataset,
    calculate_sequence_weights,
    create_multiple_loaders,
)
from protnote.data.structural_datasets import (
    ProteinStructureDataset,
    create_structural_loaders,
)
from protnote.models.egnn import StructuralProteinEncoder
from protnote.models.protein_encoders import ProteInfer
from protnote.models.ProtNote import ProtNote
from protnote.models.ProtNoteTrainer import ProtNoteTrainer
from protnote.utils.configs import get_project_root, get_setup
from protnote.utils.data import log_gpu_memory_usage, read_json, seed_everything, write_json
from protnote.utils.evaluation import EvalMetrics
from protnote.utils.losses import get_loss
from protnote.utils.main_utils import validate_arguments
from protnote.utils.models import (
    count_parameters_by_layer,
    load_model,
    sigmoid_bias_from_prob,
)

### SETUP ###
torch.cuda.empty_cache()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    run = cfg.run
    validate_arguments(cfg)

    # TODO: If running with multiple GPUs, make sure the vocabularies and embeddings have been pre-generated (otherwise, it will be generated multiple times)

    # Distributed computing
    world_size = run.gpus * run.nodes
    if run.amlt:
        run.nr = int(os.environ["NODE_RANK"])
    else:
        os.environ["MASTER_ADDR"] = "localhost"
        if run.master_port is not None:
            os.environ["MASTER_PORT"] = str(run.master_port)
        else:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                os.environ["MASTER_PORT"] = str(s.getsockname()[1])

    mp.spawn(train_validate_test, nprocs=run.gpus, args=(cfg, world_size))


def train_validate_test(gpu, cfg, world_size):
    run = cfg.run

    # Calculate GPU rank (based on node rank and GPU rank within the node) and initialize process group
    rank = run.nr * run.gpus + gpu
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    print(
        f"{'=' * 50}\n"
        f"Initializing GPU {gpu}/{run.gpus - 1} on node {run.nr};\n"
        f"    or, gpu {rank + 1}/{world_size} for all nodes.\n"
        f"{'=' * 50}"
    )

    project_root = get_project_root()

    # Check if master process
    is_master = rank == 0

    # Unpack and process the config file
    if run.model_file:
        run.model_file = str(project_root / "data" / "models" / "ProtNote" / run.model_file)
    run.save_val_test_metrics_file = str(project_root / "outputs" / "results" / run.save_val_test_metrics_file)
    task = run.annotations_path_name.split("_")[0]
    config = get_setup(cfg, is_master=is_master)
    params, paths, timestamp, logger = (
        config["params"],
        config["paths"],
        config["timestamp"],
        config["logger"],
    )

    # Set the GPU device, if using
    torch.cuda.set_device(rank)
    device = torch.device("cuda:" + str(rank) if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Seed everything so we don't go crazy
    seed_everything(params["SEED"], device)

    # Initialize W&B, if using
    if is_master and run.wandb_project is not None:
        wandb.init(
            project=run.wandb_project,
            name=f"{run.name}_{timestamp}",
            config=OmegaConf.to_container(params, resolve=True)
            | dict(
                name=run.name,
                amlt=run.amlt,
                model_file=run.model_file,
            ),
            sync_tensorboard=False,
            entity=run.wandb_entity,
        )

        if run.amlt and run.mlflow:
            import mlflow

            # MLFlow logging for Hyperdrive
            mlflow.autolog()
            mlflow.start_run()

        # Log the wandb link
        logger.info(f"W&B link: {wandb.run.get_url()}")

    # Log the params
    logger.info(OmegaConf.to_yaml(params))

    # Initialize label tokenizer
    label_tokenizer = AutoTokenizer.from_pretrained(params["LABEL_ENCODER_CHECKPOINT"], force_download=True)

    # Initialize label encoder
    label_encoder = AutoModel.from_pretrained(params["LABEL_ENCODER_CHECKPOINT"], force_download=True)
    if params["GRADIENT_CHECKPOINTING"]:
        raise NotImplementedError("Gradient checkpointing is not yet implemented.")

    # ---------------------- DATASETS ----------------------#
    # Determine encoder mode: hybrid (default) or sequence (legacy ProteInfer)
    use_hybrid = not run.use_sequence_encoder

    # Load graph index if using hybrid encoder
    graph_index = {}
    graph_dir = None
    graph_archive_path = None
    if use_hybrid:
        graph_dir = paths.get("PROCESSED_GRAPH_DIR", "")
        graph_index_path = paths.get("GRAPH_INDEX_PATH", "")
        if os.path.exists(graph_index_path):
            import json as _json

            with open(graph_index_path) as f:
                graph_index = _json.load(f)
            logger.info(f"Loaded graph index with {len(graph_index)} entries from {graph_index_path}")
        else:
            logger.warning(f"Graph index not found at {graph_index_path}. Structure data will use fallbacks.")

        # Detect archive file (single .pngrph or sharded directory)
        graph_archive_path = paths.get("GRAPH_ARCHIVE_PATH", "")
        if not graph_archive_path or not os.path.exists(graph_archive_path):
            default_archive = os.path.join(graph_dir, "graphs.pngrph") if graph_dir else ""
            if default_archive and os.path.exists(default_archive):
                graph_archive_path = default_archive
            elif graph_dir and any(Path(graph_dir).glob("graphs.shard-*-of-*.pngrph")):
                graph_archive_path = graph_dir  # open_archive() handles directory
            else:
                graph_archive_path = None
        if graph_archive_path:
            logger.info(f"Using graph archive: {graph_archive_path}")

    # Select dataset class
    DatasetClass = ProteinStructureDataset if use_hybrid else ProteinDataset

    def _make_dataset(data_paths, require_label_idxs):
        if use_hybrid:
            # ProteinStructureDataset doesn't support require_label_idxs
            kwargs = dict(
                data_paths=data_paths,
                config=config,
                logger=logger,
                label_tokenizer=label_tokenizer,
                graph_dir=graph_dir,
                graph_index=graph_index,
                graph_archive_path=graph_archive_path,
                use_atom_level=True,
            )
        else:
            kwargs = dict(
                data_paths=data_paths,
                config=config,
                logger=logger,
                require_label_idxs=require_label_idxs,
                label_tokenizer=label_tokenizer,
            )
        return DatasetClass(**kwargs)

    # Create individual datasets
    train_dataset = _make_dataset(config["dataset_paths"]["train"][0], params["GRID_SAMPLER"]) if run.train_path_name is not None else None

    validation_dataset = _make_dataset(config["dataset_paths"]["validation"][0], False) if run.validation_path_name is not None else None

    test_dataset = _make_dataset(config["dataset_paths"]["test"][0], False) if run.test_paths_names is not None else None

    # Add datasets to a dictionary
    # TODO: This does not support multiple datasets. But I think we should remove that support anyway. Too complicated.
    datasets = {
        "train": [train_dataset],
        "validation": [validation_dataset],
        "test": [test_dataset],
    }

    # Remove empty datasets. May happen in cases like only validating a model.
    datasets = {k: v for k, v in datasets.items() if v[0] is not None}

    # -----------------------------------------------------#

    # Initialize new run
    logger.info(f"################## {timestamp} RUNNING main.py ##################")

    # Define label sample sizes for train, validation, and test loaders
    label_sample_sizes = {
        "train": params["TRAIN_LABEL_SAMPLE_SIZE"],
        "validation": params["VALIDATION_LABEL_SAMPLE_SIZE"],
        "test": None,  # No sampling for the test set
    }

    # Calculate the weighting for the train dataset
    sequence_weights = None
    if params["WEIGHTED_SAMPLING"] and (run.train_path_name is not None):
        # Calculate label weights (need dict format for calculate_sequence_weights)
        logger.info("Calculating label weights for weighted sampling...")
        label_weights = datasets["train"][0].calculate_label_weights(
            power=params["INV_FREQUENCY_POWER"],
            return_list=False,  # Need dict format for calculate_sequence_weights
        )

        # Calculate sequence weights
        logger.info("Calculating sequence weights based on the label weights...")
        sequence_weights = calculate_sequence_weights(
            data=datasets["train"][0].data,
            label_inv_freq=label_weights,
            aggregation=params["SEQUENCE_WEIGHT_AGG"],
        )

        # If using clamping, clamp the weights based on the hyperparameters
        if params["SAMPLING_LOWER_CLAMP_BOUND"] is not None:
            sequence_weights = [max(x, params["SAMPLING_LOWER_CLAMP_BOUND"]) for x in sequence_weights]
        if params["SAMPLING_UPPER_CLAMP_BOUND"] is not None:
            sequence_weights = [min(x, params["SAMPLING_UPPER_CLAMP_BOUND"]) for x in sequence_weights]

    logger.info("Initializing data loaders...")
    # Define data loaders - use appropriate loader factory based on encoder type
    if use_hybrid:
        loaders = create_structural_loaders(
            datasets,
            params,
            label_sample_sizes=label_sample_sizes,
            shuffle_labels=params["SHUFFLE_LABELS"],
            in_batch_sampling=params["IN_BATCH_SAMPLING"],
            num_workers=params["NUM_WORKERS"],
            world_size=world_size,
            rank=rank,
            sequence_weights=sequence_weights,
        )
    else:
        loaders = create_multiple_loaders(
            datasets,
            params,
            label_sample_sizes=label_sample_sizes,
            shuffle_labels=params["SHUFFLE_LABELS"],
            in_batch_sampling=params["IN_BATCH_SAMPLING"],
            grid_sampler=params["GRID_SAMPLER"],
            num_workers=params["NUM_WORKERS"],
            world_size=world_size,
            rank=rank,
            sequence_weights=sequence_weights,
        )

    # Initialize protein encoder (ProteInfer or StructuralProteinEncoder)
    sequence_encoder = None

    if use_hybrid:
        # Structural encoder: ESM-C + EGNN
        sequence_encoder = StructuralProteinEncoder(
            plm_embedding_dim=params.get("ESMC_EMBEDDING_DIM", 960),  # ESM-C dim only; atom-type added internally
            protein_embedding_dim=params["PROTEIN_EMBEDDING_DIM"],
            hidden_nf=params.get("EGNN_HIDDEN_DIM", 256),
            num_layers=params.get("EGNN_N_LAYERS", 4),
            atom_type_dim=37,
            use_atom_level=True,
        )
        logger.info(f"Using structural encoder: ESM-C + EGNN (output_dim={params['PROTEIN_EMBEDDING_DIM']})")
    else:
        # Legacy ProteInfer sequence encoder
        if params["PRETRAINED_SEQUENCE_ENCODER"] and (run.model_file is None):
            sequence_encoder = ProteInfer.from_pretrained(
                weights_path=paths[f"PROTEINFER_{task}_WEIGHTS_PATH"],
                num_labels=config["embed_sequences_params"]["PROTEINFER_NUM_GO_LABELS"],
                input_channels=config["embed_sequences_params"]["INPUT_CHANNELS"],
                output_channels=config["embed_sequences_params"]["OUTPUT_CHANNELS"],
                kernel_size=config["embed_sequences_params"]["KERNEL_SIZE"],
                activation=torch.nn.ReLU,
                dilation_base=config["embed_sequences_params"]["DILATION_BASE"],
                num_resnet_blocks=config["embed_sequences_params"]["NUM_RESNET_BLOCKS"],
                bottleneck_factor=config["embed_sequences_params"]["BOTTLENECK_FACTOR"],
            )
        else:
            sequence_encoder = ProteInfer(
                num_labels=config["embed_sequences_params"]["PROTEINFER_NUM_GO_LABELS"],
                input_channels=config["embed_sequences_params"]["INPUT_CHANNELS"],
                output_channels=config["embed_sequences_params"]["OUTPUT_CHANNELS"],
                kernel_size=config["embed_sequences_params"]["KERNEL_SIZE"],
                activation=torch.nn.ReLU,
                dilation_base=config["embed_sequences_params"]["DILATION_BASE"],
                num_resnet_blocks=config["embed_sequences_params"]["NUM_RESNET_BLOCKS"],
                bottleneck_factor=config["embed_sequences_params"]["BOTTLENECK_FACTOR"],
            )
        logger.info("Using legacy ProteInfer sequence encoder")

    model = ProtNote(
        # Parameters
        protein_embedding_dim=params["PROTEIN_EMBEDDING_DIM"],
        label_embedding_dim=params["LABEL_EMBEDDING_DIM"],
        latent_dim=params["LATENT_EMBEDDING_DIM"],
        label_embedding_pooling_method=params["LABEL_EMBEDDING_POOLING_METHOD"],
        sequence_embedding_dropout=params["SEQUENCE_EMBEDDING_DROPOUT"],
        label_embedding_dropout=params["LABEL_EMBEDDING_DROPOUT"],
        label_embedding_noising_alpha=params["LABEL_EMBEDDING_NOISING_ALPHA"],
        # Encoders
        label_encoder=label_encoder,
        sequence_encoder=sequence_encoder,
        inference_descriptions_per_label=len(params["INFERENCE_GO_DESCRIPTIONS"].split("+")),
        # Output Layer
        output_mlp_hidden_dim_scale_factor=params["OUTPUT_MLP_HIDDEN_DIM_SCALE_FACTOR"],
        output_mlp_num_layers=params["OUTPUT_MLP_NUM_LAYERS"],
        output_neuron_bias=sigmoid_bias_from_prob(params["OUTPUT_NEURON_PROBABILITY_BIAS"])
        if params["OUTPUT_NEURON_PROBABILITY_BIAS"] is not None
        else None,
        outout_mlp_add_batchnorm=params["OUTPUT_MLP_BATCHNORM"],
        residual_connection=params["RESIDUAL_CONNECTION"],
        projection_head_num_layers=params["PROJECTION_HEAD_NUM_LAYERS"],
        dropout=params["OUTPUT_MLP_DROPOUT"],
        projection_head_hidden_dim_scale_factor=params["PROJECTION_HEAD_HIDDEN_DIM_SCALE_FACTOR"],
        # Training options
        label_encoder_num_trainable_layers=params["LABEL_ENCODER_NUM_TRAINABLE_LAYERS"],
        train_sequence_encoder=params["TRAIN_PROTEIN_ENCODER"] if use_hybrid else params["TRAIN_SEQUENCE_ENCODER"],
        # Batch size limits
        label_batch_size_limit=params["LABEL_BATCH_SIZE_LIMIT_NO_GRAD"],
        sequence_batch_size_limit=params["SEQUENCE_BATCH_SIZE_LIMIT_NO_GRAD"],
        # Others
        feature_fusion=config["params"]["FEATURE_FUSION"],
        temperature=config["params"]["SUPCON_TEMP"],
    )

    # Wrap the model in DDP for distributed computing
    if config["params"]["SYNC_BN"]:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = DDP(model.to(rank), device_ids=[rank], find_unused_parameters=True)

    # Calculate bce_pos_weight based on the training set
    if (params["BCE_POS_WEIGHT"] is None) and (run.train_path_name is not None):
        bce_pos_weight = datasets["train"][0].calculate_pos_weight().to(device)
    elif params["BCE_POS_WEIGHT"] is not None:
        bce_pos_weight = torch.tensor(params["BCE_POS_WEIGHT"]).to(device)
    elif params["LOSS_FN"] in ("BCE", "WeightedBCE"):
        raise ValueError("BCE_POS_WEIGHT is not provided and no training set is provided to calculate it.")
    else:
        bce_pos_weight = None

    if params["LOSS_FN"] == "WeightedBCE":
        if run.train_path_name is not None:
            logger.info("calculating WEIGHTED BCE WEIGHTS")
            label_weights = (
                datasets["train"][0]
                .calculate_label_weights(
                    inv_freq=True,
                    normalize=True,
                    return_list=True,
                    power=params["INV_FREQUENCY_POWER"],
                )
                .to(device)
            )
        else:
            raise ValueError("Must provde training set")

    elif params["LOSS_FN"] == "CBLoss":
        if run.train_path_name is not None:
            label_weights = (
                datasets["train"][0]
                .calculate_label_weights(
                    inv_freq=False,
                    normalize=False,
                    return_list=True,
                    power=params["INV_FREQUENCY_POWER"],
                )
                .to(device)
            )
        else:
            raise ValueError("Must provde training set")
    else:
        label_weights = None

    loss_fn = get_loss(config=config, bce_pos_weight=bce_pos_weight, label_weights=label_weights)

    # Initialize trainer class to handle model training, validation, and testing
    Trainer = ProtNoteTrainer(
        model=model,
        device=device,
        rank=rank,
        config=config,
        logger=logger,
        timestamp=timestamp,
        run_name=run.name,
        use_wandb=run.wandb_project is not None and is_master,
        use_amlt=run.amlt,
        loss_fn=loss_fn,
        is_master=is_master,
    )

    # Log the number of parameters by layer
    count_parameters_by_layer(model.module)

    # Load the model weights if --load-model argument is provided (using the DATA_PATH directory as the root)
    # TODO: Process model loading in the get_setup function
    if run.model_file:
        load_model(
            trainer=Trainer,
            checkpoint_path=run.model_file,
            rank=rank,
            from_checkpoint=run.from_checkpoint,
        )
        logger.info(
            f"Loading model checkpoint from {run.model_file}. If training, will continue from epoch {Trainer.epoch + 1}.\n"
        )

    # Initialize EvalMetrics
    eval_metrics = EvalMetrics(device=device)

    label_sample_sizes = {
        k: (v if v is not None else len(datasets[k][0].label_vocabulary)) for k, v in label_sample_sizes.items() if k in datasets.keys()
    }

    # Log sizes of all datasets
    [logger.info(f"{subset_name} dataset size: {len(dataset)}") for subset_name, subset in datasets.items() for dataset in subset]

    ####### TRAINING AND VALIDATION LOOPS #######
    if run.train_path_name is not None:
        # Train function
        Trainer.train(
            train_loader=loaders["train"][0],
            val_loader=loaders["validation"][0],
            train_eval_metrics=eval_metrics.get_metric_collection_with_regex(
                pattern="f1_m.*",
                threshold=0.5,
                num_labels=label_sample_sizes["train"] if not (params["IN_BATCH_SAMPLING"] or params["GRID_SAMPLER"]) else None,
            ),
            val_eval_metrics=eval_metrics.get_metric_collection_with_regex(
                pattern="f1_m.*",
                threshold=0.5,
                num_labels=label_sample_sizes["validation"],
            ),
            val_optimization_metric_name=params["OPTIMIZATION_METRIC_NAME"],
            only_represented_labels=run.eval_only_represented_labels,
        )
    else:
        logger.info("Skipping training...")

    ####### TESTING LOOP #######
    all_test_metrics = {}
    all_metrics = {}

    # Setup for validation
    run_metrics = {"name": run.name}
    if run.save_val_test_metrics and is_master:
        os.makedirs(os.path.dirname(run.save_val_test_metrics_file), exist_ok=True)
        if not os.path.exists(run.save_val_test_metrics_file):
            write_json([], run.save_val_test_metrics_file)
        metrics_results = read_json(run.save_val_test_metrics_file)

    best_th = None
    if run.validation_path_name:
        # Reinitialize the validation loader with all the data, in case we were using a subset to expedite training
        logger.info(f"\n{'=' * 100}\nTesting on validation set\n{'=' * 100}")

        # Print the batch size used
        logger.info(f"Batch size: {params['TEST_BATCH_SIZE']}")
        if is_master:
            log_gpu_memory_usage(logger, 0)

        # Final validation using all labels
        torch.cuda.empty_cache()

        if params["DECISION_TH"] is None:
            best_th, _ = Trainer.find_optimal_threshold(
                data_loader=loaders["validation"][0],
                optimization_metric_name=params["OPTIMIZATION_METRIC_NAME"],
            )

        validation_metrics = Trainer.evaluate(
            data_loader=loaders["validation"][0],  # full_val_loader,
            eval_metrics=eval_metrics.get_metric_collection_with_regex(
                pattern="f1_m.*",
                threshold=best_th if best_th is not None else params["DECISION_TH"],
                num_labels=label_sample_sizes["validation"],
            ),
            save_results=run.save_prediction_results,
            data_loader_name="final_validation",
        )
        all_metrics.update(validation_metrics)
        logger.info(json.dumps(validation_metrics, indent=4))
        if run.save_val_test_metrics:
            run_metrics.update(validation_metrics)
        logger.info("Final validation complete.")

    # Setup for testing
    if run.test_paths_names:
        for idx, test_loader in enumerate(loaders["test"]):
            logger.info(f"\n{'=' * 100}\nTesting on test set {idx + 1}/{len(loaders['test'])}\n{'=' * 100}")
            if is_master:
                log_gpu_memory_usage(logger, 0)

            # TODO: If best_val_th is not defined, alert an error to either provide a decision threshold or a validation datapath
            test_metrics = Trainer.evaluate(
                data_loader=test_loader,
                eval_metrics=eval_metrics.get_metric_collection_with_regex(
                    pattern="f1_m.*",
                    threshold=best_th if best_th is not None else params["DECISION_TH"],
                    num_labels=label_sample_sizes["test"],
                ),
                save_results=run.save_prediction_results,
                data_loader_name=f"test_{idx + 1}",
                return_embeddings=run.save_embeddings,
            )
            all_test_metrics.update(test_metrics)
            logger.info(json.dumps(test_metrics, indent=4))
            if run.save_val_test_metrics:
                run_metrics.update(test_metrics)
            logger.info("Testing complete.")

        all_metrics.update(all_test_metrics)

    ####### CLEANUP #######

    logger.info(f"\n{'=' * 100}\nTraining, validating, and testing COMPLETE\n{'=' * 100}\n")
    # W&B, MLFlow amd optional metric results saving
    if is_master:
        # Optionally save val/test results in json
        if run.save_val_test_metrics:
            metrics_results.append(run_metrics)
            write_json(metrics_results, run.save_val_test_metrics_file)
        # Log test metrics
        if run.test_paths_names:
            if run.wandb_project is not None:
                wandb.log(all_test_metrics)
            if run.amlt and run.mlflow:
                mlflow.log_metrics(all_test_metrics)

        # Log val metrics
        if run.validation_path_name:
            if run.wandb_project is not None:
                wandb.log(validation_metrics)
            if run.amlt and run.mlflow:
                mlflow.log_metrics(validation_metrics)

        # Close metric loggers
        if run.wandb_project is not None:
            wandb.finish()
        if run.amlt and run.mlflow:
            mlflow.end_run()

    # Loggers
    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()
    # Torch
    torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
