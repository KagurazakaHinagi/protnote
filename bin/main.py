import argparse
import json
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModel, AutoTokenizer

import wandb
from protnote.data.collators import collate_structure_batch
from protnote.data.datasets import (
    ProteinDataset,
    StructureProteinDataset,
    calculate_sequence_weights,
    create_multiple_loaders,
)
from protnote.models.hybrid_encoders import HybridProteinEncoder
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


def main():
    # ---------------------- HANDLE ARGUMENTS ----------------------#
    parser = argparse.ArgumentParser(description="Train and/or Test the ProtNote model.")
    parser.add_argument(
        "--train-path-name",
        type=str,
        default=None,
        help="Specify the desired train path name to train the model using names from config file. If not provided, model will not be trained. If provided, must also provide --val-path.",
    )

    parser.add_argument(
        "--validation-path-name",
        type=str,
        default=None,
        help="Specify the desired val path name to validate the model during training using names from config file. If not provided, model will not be trained. If provided, must also provide --train-path.",
    )

    parser.add_argument(
        "--full-path-name",
        type=str,
        default=None,
        help="Specify the desired full path name to define the vocabularies. Defaults to the full path name in the config file.",
    )

    parser.add_argument(
        "--test-paths-names",
        nargs="+",
        type=str,
        default=None,
        help="Specify all the desired test paths names to test the model using names from config file to test. If not provided, model will not be tested.",
    )

    parser.add_argument(
        "--annotations-path-name",
        type=str,
        default="GO_ANNOTATIONS_PATH",
        help="Name of the annotation path. Defaults to GO.",
    )

    parser.add_argument(
        "--base-label-embedding-name",
        type=str,
        default="GO_BASE_LABEL_EMBEDDING_PATH",
        help="Name of the base label embedding path. Defaults to GO.",
    )

    parser.add_argument(
        "--use-wandb",
        type=str,
        default=None,
        help="Weights & Biases project name for logging. Default is None.",
    )

    parser.add_argument(
        "--from-checkpoint",
        action="store_true",
        default=False,
        help="Continue training from a previous model checkpoint (including optimizer state and epoch). Default is False.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="ProtNote",
        help="Name of the W&B run. If not provided, a name will be generated.",
    )

    parser.add_argument(
        "--amlt",
        action="store_true",
        default=False,
        help="Run job on Amulet. Default is False.",
    )

    parser.add_argument(
        "--mlflow",
        action="store_true",
        default=False,
        help="Use MLFlow. Default is False.",
    )
    parser.add_argument("--override", nargs="*", help="Override config parameters in key-value pairs.")

    parser.add_argument(
        "--save-prediction-results",
        action="store_true",
        default=False,
        help="Save predictions and ground truth dataframe for validation and/or test",
    )

    parser.add_argument(
        "--eval-only-represented-labels",
        action="store_true",
        default=False,
        help="Evaluate only the represented labels",
    )

    parser.add_argument(
        "--save-val-test-metrics",
        action="store_true",
        default=False,
        help="Append val/test metrics to json",
    )

    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        default=False,
        help="Save different embeddings from the model FOR TEST SET ONLY",
    )

    parser.add_argument(
        "-n",
        "--nodes",
        default=1,
        type=int,
        metavar="N",
        help="Number of nodes (default: 1)",
    )

    parser.add_argument("-g", "--gpus", default=1, type=int, help="Number of gpus per node (default: 1)")

    parser.add_argument("-nr", "--nr", default=0, type=int, help="Ranking within the nodes")

    parser.add_argument(
        "--model-file",
        type=str,
        default=None,
        help=".pt weights to initialize protnote. If not provided, a new model will be initialized.",
    )

    parser.add_argument(
        "--save-val-test-metrics-file", help="json file name to append val/test metrics", type=str, default="val_test_metrics.json"
    )

    parser.add_argument(
        "--use-hybrid-encoder", action="store_true", default=False, help="Use hybrid ESM-C + EGNN encoder instead of ProteInfer CNN."
    )

    args = parser.parse_args()
    validate_arguments(args, parser)

    # TODO: If running with multiple GPUs, make sure the vocabularies and embeddings have been pre-generated (otherwise, it will be generated multiple times)

    # Distributed computing
    args.world_size = args.gpus * args.nodes
    if args.amlt:
        # os.environ['MASTER_ADDR'] = os.environ['MASTER_IP']
        args.nr = int(os.environ["NODE_RANK"])
    else:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "8889"

    mp.spawn(train_validate_test, nprocs=args.gpus, args=(args,))


def train_validate_test(gpu, args):
    # Calculate GPU rank (based on node rank and GPU rank within the node) and initialize process group
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend="nccl", init_method="env://", world_size=args.world_size, rank=rank)
    print(
        f"{'=' * 50}\n"
        f"Initializing GPU {gpu}/{args.gpus - 1} on node {args.nr};\n"
        f"    or, gpu {rank + 1}/{args.world_size} for all nodes.\n"
        f"{'=' * 50}"
    )

    project_root = get_project_root()

    # Check if master process
    is_master = rank == 0

    # Unpack and process the config file
    if args.model_file:
        args.model_file = project_root / "data" / "models" / "ProtNote" / args.model_file
    args.save_val_test_metrics_file = project_root / "outputs" / "results" / args.save_val_test_metrics_file
    task = args.annotations_path_name.split("_")[0]
    config = get_setup(
        config_path=project_root / "configs" / "base_config.yaml",
        run_name=args.name,
        overrides=args.override,
        train_path_name=args.train_path_name,
        val_path_name=args.validation_path_name,
        test_paths_names=args.test_paths_names,
        annotations_path_name=args.annotations_path_name,
        base_label_embedding_name=args.base_label_embedding_name,
        amlt=args.amlt,
        is_master=is_master,
    )
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
    if is_master and args.use_wandb is not None:
        wandb.init(
            project=args.use_wandb,
            name=f"{args.name}_{timestamp}",
            config={**params, **vars(args)},
            sync_tensorboard=False,
            entity="hinagi",
        )

        if args.amlt & args.mlflow:
            import mlflow

            # MLFlow logging for Hyperdrive
            mlflow.autolog()
            mlflow.start_run()

        # Log the wandb link
        logger.info(f"W&B link: {wandb.run.get_url()}")

    # Log the params
    logger.info(json.dumps(params, indent=4))

    # Initialize label tokenizer
    label_tokenizer = AutoTokenizer.from_pretrained(params["LABEL_ENCODER_CHECKPOINT"], force_download=True)

    # Initialize label encoder
    label_encoder = AutoModel.from_pretrained(params["LABEL_ENCODER_CHECKPOINT"], force_download=True)
    if params["GRADIENT_CHECKPOINTING"]:
        raise NotImplementedError("Gradient checkpointing is not yet implemented.")

    # ---------------------- DATASETS ----------------------#
    # Load graph index if using hybrid encoder
    use_hybrid = args.use_hybrid_encoder
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

        # Detect archive file
        graph_archive_path = paths.get("GRAPH_ARCHIVE_PATH", "")
        if not graph_archive_path or not os.path.exists(graph_archive_path):
            default_archive = os.path.join(graph_dir, "graphs.pngrph") if graph_dir else ""
            graph_archive_path = default_archive if os.path.exists(default_archive) else None
        if graph_archive_path:
            logger.info(f"Using graph archive: {graph_archive_path}")

    # Select dataset class
    DatasetClass = StructureProteinDataset if use_hybrid else ProteinDataset

    def _make_dataset(data_paths, require_label_idxs):
        kwargs = dict(
            data_paths=data_paths,
            config=config,
            logger=logger,
            require_label_idxs=require_label_idxs,
            label_tokenizer=label_tokenizer,
        )
        if use_hybrid:
            kwargs["graph_dir"] = graph_dir
            kwargs["graph_index"] = graph_index
            kwargs["graph_archive_path"] = graph_archive_path
        return DatasetClass(**kwargs)

    # Create individual datasets
    train_dataset = _make_dataset(config["dataset_paths"]["train"][0], params["GRID_SAMPLER"]) if args.train_path_name is not None else None

    validation_dataset = _make_dataset(config["dataset_paths"]["validation"][0], False) if args.validation_path_name is not None else None

    test_dataset = _make_dataset(config["dataset_paths"]["test"][0], False) if args.test_paths_names is not None else None

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
    if params["WEIGHTED_SAMPLING"] & (args.train_path_name is not None):
        # Calculate label weights
        logger.info("Calculating label weights for weighted sampling...")
        label_weights = datasets["train"][0].calculate_label_weights(power=params["INV_FREQUENCY_POWER"])

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
    # Define data loaders
    loaders = create_multiple_loaders(
        datasets,
        params,
        label_sample_sizes=label_sample_sizes,
        shuffle_labels=params["SHUFFLE_LABELS"],
        in_batch_sampling=params["IN_BATCH_SAMPLING"],
        grid_sampler=params["GRID_SAMPLER"],
        num_workers=params["NUM_WORKERS"],
        world_size=args.world_size,
        rank=rank,
        sequence_weights=sequence_weights,
        collate_fn_override=collate_structure_batch if use_hybrid else None,
    )

    # Initialize ProteInfer
    if params["PRETRAINED_SEQUENCE_ENCODER"] & (args.model_file is None):
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

    # Initialize protein encoder (ProteInfer or Hybrid)
    sequence_encoder = None
    hybrid_encoder = None
    encoder_type = "proteinfer"

    if use_hybrid:
        encoder_type = "hybrid"
        use_plm = params.get("USE_PLM_EMBEDDINGS", True)
        hybrid_encoder = HybridProteinEncoder(
            esmc_embedding_dim=params.get("ESMC_EMBEDDING_DIM", 960),
            atom_type_dim=37,
            egnn_hidden_dim=params.get("EGNN_HIDDEN_DIM", 256),
            egnn_out_dim=params.get("EGNN_OUT_DIM", 256),
            egnn_n_layers=params.get("EGNN_N_LAYERS", 4),
            output_dim=params["PROTEIN_EMBEDDING_DIM"],
            use_plm=use_plm,
            learned_aa_embedding_dim=params.get("LEARNED_AA_EMBEDDING_DIM", 128),
        )
        plm_mode = "ESM-C" if use_plm else f"learned AA embedding (dim={params.get('LEARNED_AA_EMBEDDING_DIM', 128)})"
        logger.info(f"Using hybrid encoder: {plm_mode} + EGNN (output_dim={params['PROTEIN_EMBEDDING_DIM']})")
    else:
        if params["PRETRAINED_SEQUENCE_ENCODER"] & (args.model_file is None):
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
        hybrid_encoder=hybrid_encoder,
        encoder_type=encoder_type,
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
        train_sequence_encoder=params["TRAIN_SEQUENCE_ENCODER"],
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
    if (params["BCE_POS_WEIGHT"] is None) & (args.train_path_name is not None):
        bce_pos_weight = datasets["train"][0].calculate_pos_weight().to(device)
    elif params["BCE_POS_WEIGHT"] is not None:
        bce_pos_weight = torch.tensor(params["BCE_POS_WEIGHT"]).to(device)
    else:
        raise ValueError("BCE_POS_WEIGHT is not provided and no training set is provided to calculate it.")

    if params["LOSS_FN"] == "WeightedBCE":
        if args.train_path_name is not None:
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
        if args.train_path_name is not None:
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
        run_name=args.name,
        use_wandb=args.use_wandb is not None and is_master,
        use_amlt=args.amlt,
        loss_fn=loss_fn,
        is_master=is_master,
    )

    # Log the number of parameters by layer
    count_parameters_by_layer(model.module)

    # Load the model weights if --load-model argument is provided (using the DATA_PATH directory as the root)
    # TODO: Process model loading in the get_setup function
    if args.model_file:
        load_model(
            trainer=Trainer,
            checkpoint_path=os.path.join(config["DATA_PATH"], args.model_file),
            rank=rank,
            from_checkpoint=args.from_checkpoint,
        )
        logger.info(
            f"Loading model checkpoing from {os.path.join(config['DATA_PATH'], args.model_file)}. If training, will continue from epoch {Trainer.epoch + 1}.\n"
        )

    # Initialize EvalMetrics
    eval_metrics = EvalMetrics(device=device)

    label_sample_sizes = {
        k: (v if v is not None else len(datasets[k][0].label_vocabulary)) for k, v in label_sample_sizes.items() if k in datasets.keys()
    }

    # Log sizes of all datasets
    [logger.info(f"{subset_name} dataset size: {len(dataset)}") for subset_name, subset in datasets.items() for dataset in subset]

    ####### TRAINING AND VALIDATION LOOPS #######
    if args.train_path_name is not None:
        # Train function
        Trainer.train(
            train_loader=loaders["train"][0],
            val_loader=loaders["validation"][0],
            train_eval_metrics=eval_metrics.get_metric_collection_with_regex(
                pattern="f1_m.*",
                threshold=0.5,
                num_labels=label_sample_sizes["train"] if (params["IN_BATCH_SAMPLING"] or params["GRID_SAMPLER"]) is False else None,
            ),
            val_eval_metrics=eval_metrics.get_metric_collection_with_regex(
                pattern="f1_m.*",
                threshold=0.5,
                num_labels=label_sample_sizes["validation"],
            ),
            val_optimization_metric_name=params["OPTIMIZATION_METRIC_NAME"],
            only_represented_labels=args.eval_only_represented_labels,
        )
    else:
        logger.info("Skipping training...")

    ####### TESTING LOOP #######
    all_test_metrics = {}
    all_metrics = {}

    # Setup for validation
    run_metrics = {"name": args.name}
    if args.save_val_test_metrics & is_master:
        if not os.path.exists(args.save_val_test_metrics_file):
            write_json([], args.save_val_test_metrics_file)
        metrics_results = read_json(args.save_val_test_metrics_file)

    best_th = None
    if args.validation_path_name:
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
            save_results=args.save_prediction_results,
            data_loader_name="final_validation",
        )
        all_metrics.update(validation_metrics)
        logger.info(json.dumps(validation_metrics, indent=4))
        if args.save_val_test_metrics:
            run_metrics.update(validation_metrics)
        logger.info("Final validation complete.")

    # Setup for testing
    if args.test_paths_names:
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
                save_results=args.save_prediction_results,
                data_loader_name=f"test_{idx + 1}",
                return_embeddings=args.save_embeddings,
            )
            all_test_metrics.update(test_metrics)
            logger.info(json.dumps(test_metrics, indent=4))
            if args.save_val_test_metrics:
                run_metrics.update(test_metrics)
            logger.info("Testing complete.")

        all_metrics.update(test_metrics)

    ####### CLEANUP #######

    logger.info(f"\n{'=' * 100}\nTraining, validating, and testing COMPLETE\n{'=' * 100}\n")
    logger.info(f"\n{'=' * 100}\nTraining, validating, and testing COMPLETE\n{'=' * 100}\n")
    # W&B, MLFlow amd optional metric results saving
    if is_master:
        # Optionally save val/test results in json
        if args.save_val_test_metrics:
            metrics_results.append(run_metrics)
            write_json(metrics_results, args.save_val_test_metrics_file)
        # Log test metrics
        if args.test_paths_names:
            if args.use_wandb is not None:
                wandb.log(all_test_metrics)
            if args.amlt & args.mlflow:
                mlflow.log_metrics(all_test_metrics)

        # Log val metrics
        if args.validation_path_name:
            if args.use_wandb is not None:
                wandb.log(validation_metrics)
            if args.amlt & args.mlflow:
                mlflow.log_metrics(validation_metrics)

        # Close metric loggers
        if args.use_wandb is not None:
            wandb.finish()
        if args.amlt & args.mlflow:
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
