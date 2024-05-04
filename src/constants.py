import os
from pathlib import Path

# Paths of project
ROOT_PATH = Path(__file__).parent.parent
DATA_PATH = os.path.join(ROOT_PATH, 'data')
VIDEOS_PATH = os.path.join(DATA_PATH, 'video')
