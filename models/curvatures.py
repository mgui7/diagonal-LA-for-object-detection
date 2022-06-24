"""Taken from https://github.com/DLR-RM/curvature
Various Fisher information matrix approximations.
Released under the GPL-3.0 License
"""

from abc import ABC, abstractmethod
from typing import Union, List, Any, Dict
import copy
from utils.utils import computeFisherSum

from numpy.linalg import inv, cholesky
import torch
from torch import Tensor
from torch.nn import Module, Sequential
import torch.nn.functional as F
from tqdm import tqdm
import sys
sys.path.append("..") 

from .utilities import get_eigenvectors, kron

class Curvature(ABC):
    """Base class for all src approximations.

    All src approximations are computed layer-wise (i.e. layer-wise independence is assumed s.t. no
    covariances between layers are computed, aka block-wise approximation) and stored in `state`.

    The src of the loss function is the matrix of 2nd-order derivatives of the loss w.r.t. the networks weights
    (i.e. the expected Hessian). It can be approximated by the expected Fisher information matrix and, under exponential
    family loss functions (like mean-squared error and cross-entropy loss) and piecewise linear activation functions
    (i.e. ReLU), becomes identical to the Fisher.

    Note:
        The aforementioned identity does not hold for the empirical Fisher, where the expectation is computed w.r.t.
        the data distribution instead of the models' output distribution. Also, because the latter is usually unknown,
        it is approximated through Monte Carlo integration using samples from a categorical distribution, initialized by
        the models' output.

    """

    def __init__(self,
                 model: Union[Module, Sequential],
                 layer_types: Union[List[str], str] = None):
        """Curvature class initializer.

        Args:
            model: Any (pre-trained) PyTorch model including all `torchvision` models.
            layer_types: Types of layers for which to compute src information. Supported are `Linear`, `Conv2d`
                         and `MultiheadAttention`. If `None`, all supported types are considered. Default: None.
        """
        self.model = model
        self.model_state = copy.deepcopy(model.state_dict())
        self.layer_types = list()
        if isinstance(layer_types, str):
            self.layer_types.append(layer_types)
        elif isinstance(layer_types, list):
            if layer_types:
                self.layer_types.extend(layer_types)
            else:
                self.layer_types.extend(['Linear', 'Conv2d', 'MultiheadAttention'])
        elif layer_types is None:
            self.layer_types.extend(['Linear', 'Conv2d', 'MultiheadAttention'])
        else:
            raise TypeError
        for _type in self.layer_types:
            assert _type in ['Linear', 'Conv2d', 'MultiheadAttention']
        self.state = dict()
        self.inv_state = dict()

    @staticmethod
    def _replace(sample: Tensor,
                 weight: Tensor,
                 bias: Tensor = None):
        """Modifies current model parameters by adding/subtracting quantity given in `sample`.

        Args:
            sample: Sampled offset from the mean dictated by the inverse src (variance).
            weight: The weights of one model layer.
            bias: The bias of one model layer. Optional.
        """
        if bias is not None:
            bias_sample = sample[:, -1].contiguous().view(*bias.shape)
            bias.data.add_(bias_sample)
            sample = sample[:, :-1]
        weight.data.add_(sample.contiguous().view(*weight.shape))

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any):
        """Abstract method to be implemented by each derived class individually."""
        raise NotImplementedError

    @abstractmethod
    def invert(self,
               add: Union[float, list, tuple] = 0.,
               multiply: Union[float, list, tuple] = 1.):
        """Abstract method to be implemented by each derived class individually. Inverts state.

        Args:
            add: This quantity times the identity is added to each src factor.
            multiply: Each factor is multiplied by this quantity.

        Returns:
            A dict of inverted factors and potentially other quantities required for sampling.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self,
               layer: Module) -> Tensor:
        """Abstract method to be implemented by each derived class individually. Samples from inverted state.

        Args:
            layer: A layer instance from the current model.

        Returns:
            A tensor with newly sampled weights for the given layer.
        """
        raise NotImplementedError

    def sample_and_replace(self):
        """Samples new model parameters and replaces old ones for selected layers, skipping all others."""
        self.model.load_state_dict(self.model_state)
        for layer in self.model.modules():
            if layer.__class__.__name__ in self.layer_types:
                if layer.__class__.__name__ in ['Linear', 'Conv2d']:
                    _sample = self.sample(layer)
                    self._replace(_sample, layer.weight, layer.bias)
                elif layer.__class__.__name__ == 'MultiheadAttention':
                    for weight, bias, layer in [(layer.in_proj_weight, layer.in_proj_bias, 'attn_in'),
                                                (layer.out_proj.weight, layer.out_proj.bias, 'attn_out')]:
                        _sample = self.sample(layer)
                        self._replace(_sample, weight, bias)


class Diagonal(Curvature):
    r"""The diagonal Fisher information or Generalized Gauss Newton matrix approximation.

    It is defined as :math:`F_{DIAG}=\mathrm{diag}(F)` with `F` being the Fisher defined in the `FISHER` class.
    Code inspired by https://github.com/wjmaddox/swa_gaussian/blob/master/swag/posteriors/diag_laplace.py.
    """

    def update(self,
               batch_size: int):
        """Computes the diagonal src for selected layer types, skipping all others.

        Args:
            batch_size: The size of the current batch.
        """
        for layer in self.model.modules():
            if layer.__class__.__name__ in self.layer_types:
                if layer.__class__.__name__ in ['Linear', 'Conv2d']:
                    grads = layer.weight.grad.contiguous().view(layer.weight.grad.shape[0], -1)
                    if layer.bias is not None:
                        grads = torch.cat([grads, layer.bias.grad.unsqueeze(dim=1)], dim=1)
                    grads = grads ** 2 * batch_size
                    if layer in self.state:
                        self.state[layer] += grads
                    else:
                        self.state[layer] = grads
                elif layer.__class__.__name__ == 'MultiheadAttention':
                    grads = layer.in_proj_weight.grad.contiguous().view(layer.in_proj_weight.grad.shape[0], -1)
                    grads = torch.cat([grads, layer.in_proj_bias.grad.unsqueeze(dim=1)], dim=1)
                    grads = grads ** 2 * batch_size
                    if 'attn_in' in self.state:
                        self.state['attn_in'] += grads
                    else:
                        self.state['attn_in'] = grads

                    grads = layer.out_proj.weight.grad.contiguous().view(layer.out_proj.weight.grad.shape[0], -1)
                    grads = torch.cat([grads, layer.out_proj.bias.grad.unsqueeze(dim=1)], dim=1)
                    grads = grads ** 2 * batch_size
                    if 'attn_out' in self.state:
                        self.state['attn_out'] += grads
                    else:
                        self.state['attn_out'] = grads

    def invert(self,
               add: Union[float, list, tuple] = 0.,
               multiply: Union[float, list, tuple] = 1.):
        assert self.state, "State dict is empty. Did you call 'update' prior to this?"
        if self.inv_state:
            Warning("State has already been inverted. Is this expected?")
        for index, (layer, value) in enumerate(self.state.items()):
            if isinstance(add, (list, tuple)) and isinstance(multiply, (list, tuple)):
                assert len(add) == len(multiply) == len(self.state)
                n, s = add[index], multiply[index]
            else:
                n, s = add, multiply
            self.inv_state[layer] = torch.reciprocal(s * value + n).sqrt()

    def sample(self,
               layer: Union[Module, str]):
        assert self.inv_state, "Inverse state dict is empty. Did you call 'invert' prior to this?"
        return self.inv_state[layer].new(self.inv_state[layer].size()).normal_() * self.inv_state[layer]
