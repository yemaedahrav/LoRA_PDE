import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import seaborn as sns
import numpy as np
import h5py
import argparse

import matplotlib.pyplot as plt
from collections import OrderedDict

sns.set_style("white")

parser = argparse.ArgumentParser(description ='Reynolds Number')
parser.add_argument('--Re', type=int, default=60, help='Reynolds Number')
args = parser.parse_args()