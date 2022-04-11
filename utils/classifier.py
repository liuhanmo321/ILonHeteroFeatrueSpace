import torch

torch.backends.cudnn.benchmark = True
import torch.nn as nn
from torch.nn import Transformer
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
# from PIL import Image
# from tqdm import tqdm
import time
import copy
import random
import pandas as pd
import math

class Specific_Model(nn.Module):
    def __init__(self, params):
        nn.Module.__init__(self)
        # general layers
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.dropout_rate = 0.1
        self.dropout = nn.Dropout(p=self.dropout_rate)

        # specific network structure

        # hyper parameters
        self.specific_input_size = params['num_specific_features']
        self.specific_hidden_size = 64
        self.specific_out_size = params['num_specific_classes']

        # MLP structure
        self.fc_specific_in = nn.Linear(self.specific_input_size, self.specific_hidden_size)
        self.fc_specific_mid = nn.Linear(self.specific_hidden_size, 32)
        self.fc_specific_out = nn.Linear(32, self.specific_out_size)

    def forward(self, x):
        '''
        here assumes two parts of input, shared and specific
        :param x_shared: shared features
        :param x_specifics: specific features. For now we assume all specific features are kept.
        :return: processed prediction
        '''

        specific_out = self.fc_specific_in(x)
        specific_out = self.dropout(specific_out)
        specific_out = self.relu(specific_out)
        specific_out = self.fc_specific_mid(specific_out)
        specific_out = self.dropout(specific_out)
        specific_out = self.relu(specific_out)
        specific_out = self.fc_specific_out(specific_out)
        # specific_out = self.softmax(specific_out)

        return specific_out 