from abc import ABC
from typing import Union, List, Any, Dict
import copy

import torch
from torch import Tensor
from torch.nn import Module, Sequential
import torch.nn.functional as F
from .fisher_tools import kf_eigens, invert_cholesky, invert_fisher, kf_inner

class KFAC(ABC):
    r"""The Kronecker-factored Fisher information matrix approximation.

    For a single datum, the Fisher can be Kronecker-factorized into two much smaller matrices `Q` and `H`, aka
    `Kronecker factors`, s.t. :math:`F=Q\otimes H` where :math:`Q=zz^T` and :math:`H=\nabla_a^2 E(W)` with `z` being the
    output vector of the previous layer, `a` the `pre-activation` of the current layer (i.e. the output of the previous
    layer before being passed through the non-linearity) and `E(W)` the loss. For the expected Fisher,
    :math:`\mathbb{E}[Q\otimes H]\approx\mathbb{E}[Q]\otimes\mathbb{E}[H]` is assumed, which might not necessarily be
    the case.

    Code adapted from <https://github.com/DLR-RM/curvature>

"""
    def __init__(self,
                 model: Union[Module, Sequential],
                 device: str,
                 layer_types: Union[List[str], str] = None):
        """KFAC class initializer.

        For the recursive computation of `H`, outputs and inputs for each layer are recorded in `record`. Forward and
        backward hook handles are stored in `hooks` for subsequent removal.

        Args:
            model: Any (pre-trained) PyTorch model including all `torchvision` models.
        """

        self.model = model
        self.model_state = copy.deepcopy(model.state_dict())
        self.layer_types = ['Linear', 'Conv2d', 'MultiheadAttention']


        self.fisher = dict()
        self.invchol = dict()
        self.invfisher = dict()
        self.eigvals = dict()
        self.eigvecs = dict()

        self.hooks = list()
        self.record = dict()
        self.grad = dict()
 
        self.eigenspectrum = dict()

        self.state = dict()

        for layer in model.modules():
            if layer.__class__.__name__ in self.layer_types:
                if layer.__class__.__name__ in ['Linear', 'Conv2d']:
                    self.record[layer] = [None, None]
                    self.hooks.append(layer.register_forward_pre_hook(self._save_input))
                    self.hooks.append(layer.register_backward_hook(self._save_output))
                elif layer.__class__.__name__ == 'MultiheadAttention':
                    raise NotImplementedError

    def _save_input(self, module, input):
        self.record[module][0] = input[0]

    def _save_output(self, module, grad_input, grad_output):
        self.record[module][1] = grad_output[0] * grad_output[0].size(0)

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
        # if bias is not None:
        #     bias_sample = sample[:, -1].contiguous().view(*bias.shape)
        #     bias.data.add_(bias_sample)
        #     sample = sample[:, :-1]
        weight.data.add_(sample.contiguous().view(*weight.shape))

    def sample_and_replace(self,
                        eps: float,
                        eigen_index: int,
                        ):
        """Samples new model parameters and replaces old ones for selected layers, skipping all others."""
        self.model.load_state_dict(self.model_state)
        for name, layer in self.model.named_modules():
            if layer.__class__.__name__ in self.layer_types:
                if layer.__class__.__name__ in ['Linear', 'Conv2d']:
                    if name in self.fisher.keys():

                        if eigen_index is not None:
                            assert self.eigvals and self.eigvecs, "Eigenspectrum not calculated yet"
                            _sample = self.sample_eigen(name, eigen_index)
                        else:
                            _sample = self.sample(name)

                        _sample = self.sample(name)
                        if eps is not None:
                            _sample*= eps

                        self._replace(_sample, layer.weight, layer.bias)
                    

    def update_grad(self, log):
        """Computes Kronecker decomposition of layer-wise gradients via forward and backward vectors.
            Updates previous list of gradient vector outer decompositions
        """
        for name, layer in self.model.named_modules():
            module_class = layer.__class__.__name__
            if layer.__class__.__name__ in self.layer_types:
                if module_class in ['Linear', 'Conv2d']:
                    forward, backward = self.record[layer]                                                                                                  
                    if backward != None and forward != None:
                        if log:
                            print('FORWARD SIZE ', forward.size())
                            print('BACKWARD SIZE', backward.size())
                            print('OK gradient for layer', name ,' --> ', layer)
                        if module_class == 'Conv2d':
                            forward = F.unfold(forward, layer.kernel_size, padding=layer.padding, stride=layer.stride)
                            forward = forward.data.permute(1, 0, 2).contiguous().view(forward.shape[1], -1)
                        else:
                            forward = forward.data.t()

                        if layer.bias is not None:
                            ones = torch.ones_like(forward[:1])
                            forward = torch.cat([forward, ones], dim=0)

                        # 2nd factor: backward
                        if module_class == 'Conv2d':
                            backward = backward.data.permute(1, 0, 2, 3).contiguous().view(backward.shape[1], -1)
                        else:
                            backward = backward.data.t()

                        '''Update single gradient outer decomposition to current value'''
                        self.grad[name] = [forward.mean(dim=1, keepdim = True), backward.mean(dim=1, keepdim=True)]

                    else:
                        if log:
                            print("None gradient for layer", name , ' --> ', layer)


    def update_fisher(self, log):
        """Computes the 1st and 2nd Kronecker factor `Q` and `H` for each selected layer type, skipping all others.

        Todo: Check code against papers.

        Args:
            batch_size: The size of the current batch.
        """
        for name, layer in self.model.named_modules():
            module_class = layer.__class__.__name__
            if layer.__class__.__name__ in self.layer_types:
                if module_class in ['Linear', 'Conv2d']:
                    forward, backward = self.record[layer]

                    if backward is not None and forward is not None:
                        if log:
                            print(f'forward shape = {forward.shape}')
                            print(f'backward shape = {backward.shape}')
                            print(f'gradient OK for layer {name}')

                        # 1st factor: Q
                        if module_class == 'Conv2d':
                            forward = F.unfold(forward, layer.kernel_size, padding=layer.padding, stride=layer.stride)
                            forward = forward.data.permute(1, 0, 2).contiguous().view(forward.shape[1], -1)
                        else:
                            forward = forward.data.t()
                        if layer.bias is not None:
                            ones = torch.ones_like(forward[:1])
                            forward = torch.cat([forward, ones], dim=0)
                        first_factor = torch.mm(forward, forward.t()) / float(forward.shape[1])

                        if log:
                            print(f'Q shape = {first_factor.shape}')

                        # 2nd factor: H
                        if module_class == 'Conv2d':
                            backward = backward.data.permute(1, 0, 2, 3).contiguous().view(backward.shape[1], -1)
                        else:
                            backward = backward.data.t()
                        second_factor = torch.mm(backward, backward.t()) / float(backward.shape[1])
                        if log:
                            print(f'H shape = {second_factor.shape}')
                            print('--------------------' + '\n')

                        # Expectation
                        if name in self.fisher:
                            self.fisher[name][0] += first_factor
                            self.fisher[name][1] += second_factor
                        else:
                            self.fisher[name] = [first_factor, second_factor]
                    else:
                        if log:
                            print(f'None gradient for layer {name}')


    def invert_cholesky(self,
                        ):
        """Compute inverse cholesky of each fisher (Q,H) component """
        
        assert self.fisher, "fisher is empty. Did you call 'update' prior to this?"
        assert self.eigvals

        print(f"Computing inverted Cholesky of fisher ...")

        self.invchol = invert_cholesky(self.fisher, self.eigvals)


    def invert_fisher(self):
        """Compute inverse of each fisher (Q,H) component """
        
        assert self.fisher, "fisher is empty. Did you call 'update' prior to this?"

        if not self.invchol:
            self.invchol = invert_cholesky(self.fisher)
            self.invfisher = invert_fisher(self.invchol)
            
        else:
            self.invfisher = invert_fisher(self.invchol)


    
    def kf_inner(self, grad_1, grad_2):
        r'''
        Riemann metric of curved space of NNs
        AT a given point of the neural manifold (trained model: self.model) computes the inner-product of tangent vectors (gradients) according to the curvature 
        '''
        assert self.fisher, "fisher state is empty"
        assert self.invfisher, "fisher is not inverted"

        return kf_inner(grad_1, grad_2, self.invfisher)

 
    def kf_eigens(self):
        r"""
        Calculate layer-wise eigen-spectrum of the KFAC factors
        """
        print("Calculating eigenvalues of the fisher")
        eigvals, eigvecs = kf_eigens(self.fisher)
        
        self.eigvals, self.eigvecs = eigvals, eigvecs
        

    def sample(self,
               name: str) -> Tensor:
        assert self.invchol, "Inverse Cholesky state dict is empty. Did you call 'invert' prior to this?"

        first, second = self.invchol[name]
        z = torch.randn(first.size(0), second.size(0), device=first.device, dtype=first.dtype)
        sample = (first @ z @ second.t()).t()  # Final transpose because PyTorch uses channels first
        return sample
    
    
    def select_eigen(self,
                     name: str,
                     eigen_index: Union[int, float],
                     ) -> Tensor:
        """
        Select (deterministic) weights along KF eigen-directions, scaled by eigenvalue
        """
        xxt_eigvecs, ggt_eigvecs = self.eigvecs[name]
        xxt_eigvals, ggt_eigvals = self.eigvals[name]
        lambdas = torch.outer(xxt_eigvals, ggt_eigvals)

        if isinstance(eigen_index, int):
            _lambda = lambdas.view(-1)[eigen_index]
        elif isinstance(eigen_index, float):
            assert 0. <= eigen_index <= 1., 'float eigen_index out of range, expecting in range [0,1]'
            idx = int(eigen_index*(len(lambdas.view(-1)) -1))
            _lambda = lambdas.view(-1)[idx]
        else:
            raise ValueError("wrong data type for eigen_index")
        
        q_idx, h_idx = (lambdas==_lambda).nonzero()[0]

        return torch.outer(xxt_eigvecs[:, q_idx], ggt_eigvecs[:, h_idx]).t()
    

    def sample_eigen(self,
                     name: str,
                     eigen_index: Union[int, float],
                     ) -> Tensor:
        """
        Sample weights from KF posterior and project to eigenspaces
        """
        sample = self.sample(name = name).t()

        xxt_eigvecs, ggt_eigvecs = self.eigvecs[name]
        xxt_eigvals, ggt_eigvals = self.eigvals[name]
        lambdas = torch.outer(xxt_eigvals, ggt_eigvals)

        if isinstance(eigen_index, int):
            _lambda = lambdas.view(-1)[eigen_index]
        elif isinstance(eigen_index, float):
            assert 0. <= eigen_index <= 1., 'float eigen_index out of range, expecting in range [0,1]'
            idx = int(eigen_index*(len(lambdas.view(-1)) -1))
            _lambda = lambdas.view(-1)[idx]
        else:
            raise ValueError("wrong data type for eigen_index")
        
        q_idx, h_idx = (lambdas==_lambda).nonzer()[0]

        q_proj = xxt_eigvecs[:, [q_idx]] @ xxt_eigvecs[:, [q_idx]].t()
        h_proj = ggt_eigvecs[:, [h_idx]] @ ggt_eigvecs[:, [h_idx]].t()

        return (q_proj @ sample @ h_proj.t()).t()
