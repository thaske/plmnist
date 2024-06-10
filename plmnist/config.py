import os
from pathlib import Path

repo_root = Path(__file__).parent.parent

DATA_PATH = str(os.environ.get("PLM_DATA_PATH", repo_root / "data"))
LOG_PATH = str(os.environ.get("PLM_LOG_PATH", repo_root / "logs"))
RESULT_PATH = str(os.environ.get("PLM_RESULT_PATH", repo_root / "results"))
NUM_EPOCHS = int(os.environ.get("PLM_NUM_EPOCHS", 3))
BATCH_SIZE = int(os.environ.get("PLM_BATCH_SIZE", 256))
HIDDEN_SIZE = int(os.environ.get("PLM_HIDDEN_SIZE", 64))
LEARNING_RATE = float(os.environ.get("PLM_LEARNING_RATE", 2e-4))
DROPOUT_PROB = float(os.environ.get("PLM_DROPOUT_PROB", 0.1))
FGSM_EPSILON = float(os.environ.get("PLM_FGSM_EPSILON", 0.05))
SEED = int(os.environ.get("PLM_SEED", 42))
