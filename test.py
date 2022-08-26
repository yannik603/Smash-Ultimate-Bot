import numpy as np
from tensorboard import summary
import torch
from torch import nn
from torchsummary import summary
##make a random np array with 3 dimensions



online = nn.Sequential(
#make a model that takes in an input of 3x48x48 and outputs one of 13 classes
        nn.Conv2d(3, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=4, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(4096, 2024),
        nn.ReLU(),
        nn.Linear(2024, 500),
        nn.ReLU(),
        nn.Linear(500, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 21))

#print(online)
summary(online, (3, 100, 100))