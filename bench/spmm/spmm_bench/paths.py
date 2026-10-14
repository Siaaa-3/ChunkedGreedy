"""Project paths (SpMM_methods is vendored in the repository)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
CURRENT_RUN_FILE = PROJECT_ROOT / "current_run.txt"
SPMM_METHODS = PROJECT_ROOT / "SpMM_methods"

ACC_ROOT = SPMM_METHODS / "AccSpMM"
DTC_ROOT = SPMM_METHODS / "DTC-SpMM_ASPLOS24"
FLASH_ROOT = SPMM_METHODS / "FlashSparse" / "FlashSparse"
VOLTRIX_ROOT = SPMM_METHODS / "Voltrix-SpMM"
CLUSTERWISE_ROOT = SPMM_METHODS / "clusterwise-spgemm-main"
