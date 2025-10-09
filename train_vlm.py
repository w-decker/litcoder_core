from encoding.assembly.assembly_loader import load_assembly
from encoding.features.factory import FeatureExtractorFactory
from encoding.downsample.downsampling import Downsampler
from encoding.models.nested_cv import NestedCVModel
from encoding.trainer import AbstractTrainer
import logging

import os
os.environ['TRANSFORMERS_CACHE'] = "/storage/home/hcoda1/4/jdecker37/scratch/transformers_cache"


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 1) Load prepackaged assembly
assembly_path = "assembly_lebel_uts03.pkl"
assembly = load_assembly(assembly_path)

config = {
    "model_name": "Qwen/Qwen2-VL-2B-Instruct",
    "layer_idx": 9,
    "device": "cuda"
}

# 2) Configure components (wordrate-only)
extractor = FeatureExtractorFactory.create_extractor(
    modality="vision_language_model",
    model_name="Qwen2-VL-2B-Instruct",
    config=config,
    cache_dir="cache",
)


downsampler = Downsampler()
model = NestedCVModel(model_name="ridge_regression")

# FIR, downsampling, and trimming match our LeBel defaults
fir_delays = [1, 2, 3, 4]
trimming_config = {
    "train_features_start": 10, "train_features_end": -5,
    "train_targets_start": 0,  "train_targets_end": None,
    "test_features_start": 50,  "test_features_end": -5,
    "test_targets_start": 40,   "test_targets_end": None,
}

downsample_method = "lanczos"
downsample_config = {"window": 3, "cutoff_mult": 1.0}

# 3) Train
trainer = AbstractTrainer(
    assembly=assembly,
    feature_extractors=[extractor],
    downsampler=downsampler,
    model=model,
    fir_delays=fir_delays,
    trimming_config=trimming_config,
    use_train_test_split=True,
    logger_backend="wandb",
    wandb_project_name="lebel-vlm-qwen",
    dataset_type="lebel",
    results_dir="results",
    downsample_config=downsample_config,
)

logger.info("Starting training (wordrate only, no extra kwargs)...")
metrics = trainer.train()

logger.info("\n=== Final Results ===")
logger.info(f"Median correlation: {metrics.get('median_score', float('nan')):.4f}")
if "n_significant" in metrics:
    logger.info(f"Significant voxels: {metrics['n_significant']}")