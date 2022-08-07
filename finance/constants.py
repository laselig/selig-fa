import os
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))  # this is the project root
MAX_WORKERS = os.cpu_count()
DATA_DIR = f"{ROOT_DIR}\.data"
PLOTS_DIR = f"{ROOT_DIR}\.plots"
# print(DATA_DIR, PLOT_DIR)
