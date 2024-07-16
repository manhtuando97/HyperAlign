import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv

import random
from itertools import permutations
from typing import Optional, Callable

import numpy as np

from torch.nn import Parameter

from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_scatter import scatter_add


# Discriminator
class Discriminator(torch.nn.Module):
	def __init__(self, input_size, hidden_size2):
		super(Discriminator, self).__init__()
		self.hidden = torch.nn.Linear(input_size, hidden_size2)
		self.hidden2 = torch.nn.Linear(hidden_size2, hidden_size2)
		self.output = torch.nn.Linear(hidden_size2, 1)
	def forward(self, input_embd):
		return self.output(F.leaky_relu(self.hidden2(F.leaky_relu(self.hidden(input_embd), 0.2, inplace=True)), 0.2, inplace=True))

# Transformation Function
class transformation(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super(transformation, self).__init__()
		self.fc1 = nn.Linear(input_dim, hidden_dim)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden_dim, output_dim)

	def forward(self, x):
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		return x

# No Trans
class notrans(torch.nn.Module):
	def __init__(self):
		super(notrans, self).__init__()
	def forward(self, input_embd):
		return input_embd

# Reconstruction Function
class ReconDNN(torch.nn.Module):
	def __init__(self, hidden_size, feature_size, hidden_size2=128):
		super(ReconDNN, self).__init__()
		self.hidden = torch.nn.Linear(hidden_size, hidden_size2)
		self.output = torch.nn.Linear(hidden_size2, feature_size)
	def forward(self, input_embd):
		return self.output(F.relu(self.hidden(input_embd)))