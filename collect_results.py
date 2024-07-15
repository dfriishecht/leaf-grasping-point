import sys
sys.path.append('~')

import argparse
import glob
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
import os, os.path
HOME_DIR = os.path.expanduser('~')

def max_data_count():
    DIR = 'SDF_OUT/temp'
    img_count = len([name for name in os.listdir(HOME_DIR+'/'+DIR) 
               if os.path.isfile(os.path.join(HOME_DIR+'/'+DIR, name))])
    print(f"Total datasets to process: {img_count}")
    return img_count

