import os
from pathlib import Path

# Paths of project
ROOT_PATH = Path(__file__).parent.parent
DATA_PATH = os.path.join(ROOT_PATH, 'data')
VIDEOS_PATH = os.path.join(DATA_PATH, 'video')
VIDEO_NAME = 'Walking.54138969.mp4'
INIT_FRAME_NAME = 'init_frame_54138969.png'
BS_PATH = os.path.join(DATA_PATH, 'bs')
