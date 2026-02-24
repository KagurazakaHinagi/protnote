import datetime
import logging
import os
import sys
import time
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def get_logger():
    # Create a custom logger
    logger = logging.getLogger(__name__)

    # Set the logging level to INFO
    logger.setLevel(logging.INFO)

    # Create a console handler and set its level to INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter that includes the current date and time
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Set the formatter for the console handler
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    # Example usage
    logger.info("This is an info message.")
    return logger


def register_resolvers():
    """Register custom OmegaConf resolvers. Safe to call multiple times."""
    if not OmegaConf.has_resolver("project_root"):
        OmegaConf.register_new_resolver("project_root", lambda: str(get_project_root()))
    if not OmegaConf.has_resolver("data_root"):
        OmegaConf.register_new_resolver("data_root", lambda: _get_data_root())
    if not OmegaConf.has_resolver("output_root"):
        OmegaConf.register_new_resolver("output_root", lambda: _get_output_root())


def _get_data_root():
    if os.environ.get("AMLT_DATA_DIR"):
        return os.environ["AMLT_DATA_DIR"]
    return os.path.join(str(get_project_root()), "data")


def _get_output_root():
    if os.environ.get("AMLT_OUTPUT_DIR"):
        return os.environ["AMLT_OUTPUT_DIR"]
    return os.path.join(str(get_project_root()), "outputs")


def generate_label_embedding_path(params: dict, base_label_embedding_path: str):
    """
    Generates the name of the file that caches label embeddings. Needed due to different
    ways of pooling embeddings, different types of go descriptions and other paramters.
    This way we can store different versions/types of label embeddings for caching
    """
    assert params["LABEL_ENCODER_CHECKPOINT"] in [
        "microsoft/biogpt",
        "intfloat/e5-large-v2",
        "intfloat/multilingual-e5-large-instruct",
    ], "Model not supported"

    MODEL_NAME_2_NICKNAME = {
        "microsoft/biogpt": "BioGPT",
        "intfloat/e5-large-v2": "E5",
        "intfloat/multilingual-e5-large-instruct": "E5_multiling_inst",
    }

    label_embedding_path = base_label_embedding_path.split("/")
    temp = label_embedding_path[-1].split(".")

    base_model = temp[0].split("_")
    base_model = "_".join([base_model[0]] + [MODEL_NAME_2_NICKNAME[params["LABEL_ENCODER_CHECKPOINT"]]] + base_model[1:])

    label_embedding_path[-1] = base_model + "_" + params["LABEL_EMBEDDING_POOLING_METHOD"] + "." + temp[1]

    label_embedding_path = "/".join(label_embedding_path)
    return label_embedding_path


def resolve_paths(paths_cfg, data_root: str, output_root: str) -> dict:
    """Flatten nested paths config and prepend roots."""
    section_roots = {"data_paths": data_root, "output_paths": output_root}
    return {
        key: os.path.join(section_roots[section], value)
        for section, section_values in OmegaConf.to_container(paths_cfg, resolve=True).items()
        for key, value in section_values.items()
    }


def _build_dataset_paths(paths: dict, run) -> dict:
    """Build dataset path dicts from resolved paths and run config."""
    train_path_name = run.train_path_name
    val_path_name = run.validation_path_name
    test_paths_names = run.test_paths_names
    annotations_path_name = run.annotations_path_name

    # Normalize test_paths_names to a list (config may provide a single string)
    if isinstance(test_paths_names, str):
        test_paths_names = [test_paths_names]

    train_paths_list = (
        [
            {
                "data_path": paths[train_path_name],
                "dataset_type": "train",
                "annotations_path": paths[annotations_path_name],
            }
        ]
        if train_path_name is not None
        else []
    )

    val_paths_list = (
        [
            {
                "data_path": paths[val_path_name],
                "dataset_type": "validation",
                "annotations_path": paths[annotations_path_name],
            }
        ]
        if val_path_name is not None
        else []
    )

    test_paths_list = (
        [
            {
                "data_path": paths[key],
                "dataset_type": "test",
                "annotations_path": paths[annotations_path_name],
            }
            for key in test_paths_names
        ]
        if test_paths_names is not None
        else []
    )

    return {
        "train": train_paths_list,
        "validation": val_paths_list,
        "test": test_paths_list,
    }


def _setup_logging(paths: dict, run_name: str, is_master: bool):
    """Set up timezone, timestamp, and logging handlers."""
    os.environ["TZ"] = "US/Pacific"
    time.tzset()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S %Z").strip()

    log_dir = paths["LOG_DIR"]
    if not os.path.exists(log_dir) and is_master:
        try:
            os.makedirs(log_dir)
        except FileExistsError:
            print(f"Log directory {log_dir} already exists. is_master={is_master}")
            pass
    full_log_path = os.path.join(log_dir, f"{timestamp}_{run_name}.log")

    logger = logging.getLogger()
    # Clear existing handlers to avoid duplicates on repeated calls (e.g. DDP)
    logger.handlers.clear()

    if is_master:
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)-4s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S %Z",
        )

        file_handler = logging.FileHandler(full_log_path, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        logger.info(f"Logging to {full_log_path} and console...")
    else:
        logger.setLevel(logging.CRITICAL + 1)

    return timestamp, logger


def get_setup(
    cfg: DictConfig,
    is_master: bool = True,
) -> dict:
    """Post-process a Hydra-loaded config: resolve paths, set up logging,
    build dataset path dicts. Called after @hydra.main or compose()."""

    register_resolvers()
    run = cfg.run

    amlt = run.amlt

    # 1. Determine root paths (same AMLT logic)
    if amlt:
        ROOT_PATH = os.getcwd()
        DATA_PATH = os.environ["AMLT_DATA_DIR"]
        OUTPUT_PATH = os.environ["AMLT_OUTPUT_DIR"]
    else:
        ROOT_PATH = str(get_project_root())
        print(ROOT_PATH)
        DATA_PATH = os.path.join(ROOT_PATH, "data")
        OUTPUT_PATH = os.path.join(ROOT_PATH, "outputs")
        if not os.path.exists(OUTPUT_PATH) and is_master:
            os.makedirs(OUTPUT_PATH)

    # 2. Flatten and resolve paths
    paths = resolve_paths(cfg.paths, DATA_PATH, OUTPUT_PATH)

    # 3. Build dataset_paths
    dataset_paths = _build_dataset_paths(paths, run)

    # 4. Logging setup
    timestamp, logger = _setup_logging(paths, run.name, is_master)

    # 5. Generate label embedding path
    params_dict = OmegaConf.to_container(cfg.params, resolve=True)
    label_embedding_path = generate_label_embedding_path(
        params=params_dict,
        base_label_embedding_path=paths[run.base_label_embedding_name],
    )

    # 6. Return same dict shape as before
    return {
        "params": cfg.params,
        "embed_sequences_params": cfg.encoder.proteinfer,
        "paths": paths,
        "dataset_paths": dataset_paths,
        "remote_data": cfg.remote,
        "timestamp": timestamp,
        "logger": logger,
        "ROOT_PATH": ROOT_PATH,
        "DATA_PATH": DATA_PATH,
        "OUTPUT_PATH": OUTPUT_PATH,
        "LABEL_EMBEDDING_PATH": label_embedding_path,
    }


def get_project_root():
    """Dynamically determine the project root."""
    return Path(__file__).resolve().parent.parent.parent  # Adjust based on the folder structure


def construct_absolute_paths(dir: str, files: list) -> list:
    return [Path(dir) / file for file in files]
