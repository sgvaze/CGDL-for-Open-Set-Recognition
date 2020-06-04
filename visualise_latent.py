import torchvision
import torch

import numpy as np
import os

from torchvision import transforms
import torch.nn.functional as F

# from utils.datasets_and_loaders import SubSampleLoaderCreator, CustomCIFAR10
# from utils.open_world_dataset import DummyDataset, MNISTScale
# from models.classifier32 import Classifier32
# from utils.util_funcs import plot_grid
#
# from evaluate.get_model_funcs import counterfactual_images
#
# from torchray.attribution.common import Probe, get_module
# from torchray.attribution.grad_cam import gradient_to_grad_cam_saliency
# from torchray.benchmark import get_example_data, plot_example

from model import LVAE
from matplotlib import pyplot as plt

# Load files with means
root_path = '/users/sagar/open_world_learning/cgdl/lvae100'
val_dataset = 'omn'

train_means_file = os.path.join(root_path, 'train_fea.txt')
val_means_file = os.path.join(root_path, '{}_fea.txt'.format(val_dataset))

dummy = 0