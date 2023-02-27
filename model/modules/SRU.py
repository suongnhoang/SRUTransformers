import copy
import warnings
import math
from typing import List, Tuple, Union, Optional, Sequence, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

# from ..ops import (elementwise_recurrence_inference,
#                      elementwise_recurrence_gpu,
#                      elementwise_recurrence_naive)


class SRUCell(nn.Module):
    """
    A single SRU layer as per `LSTMCell`, `GRUCell` in Pytorch.
    """

    __constants__ = ['input_size', 'hidden_size', 'output_size', 'rnn_dropout',
                     'dropout', 'bidirectional', 'has_skip_term', 'highway_bias',
                     'v1', 'rescale', 'activation_type', 'activation', 'transform_module',
                     'projection_size', 'num_matrices', 'layer_norm',
                     'scale_x', 'normalize_after', 'weight_c_init', ]

    scale_x: Tensor

    initialized = False
    elementwise_recurrence_inference = None
    elementwise_recurrence_gpu = None
    elementwise_recurrence_naive = None

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 dropout: float = 0.0,
                 rnn_dropout: float = 0.0,
                 bidirectional: bool = False,
                 projection_size: int = 0,
                 highway_bias: float = 0.0,
                 layer_norm: bool = False,
                 normalize_after: bool = True,
                 transform_module: Optional[nn.Module] = None,
                 rescale: bool = False,
                 has_skip_term: bool = True,
                 use_tanh: bool = False,
                 v1: bool = False,
                 amp_recurrence_fp16: bool = True,
                 weight_c_init: float = 1.0):
        """Initialize the SRUCell module.

        Parameters
        ----------
        input_size: int
            the number of features in the input `x`
        hidden_size: int
            the number of features in the hidden state *for each
            direction*
        dropout: float, optional
            the dropout value applied between layers (default=0)
        rnn_dropout: float, optional
            [DEPRECATED] the variational dropout value (default=0)
            This option is deprecated because minimal performance
            improvement, and increases codebase size. This option will
            be removed at the next major version upgrade
        bidirectional: bool, optional
            if True, set the module as a bidirectional SRU
            (default=False)
        projection_size: int, optional
            if non-zero, factorize the ``weight`` parameter matrix as a
            product of two parameter matrices, using an innder dimension
            ``projection_size`` (default=0)
        use_tanh: bool, optional
            [DEPRECATED] if True, apply `tanh` activation to the hidden
            state (default=False). `tanh` is deprecated because minimal
            performance improvement, and increases codebase size. This
            option will be removed at the next major version upgrade.
        highway_bias: float, optional
            the initial value of the bias used in the highway (sigmoid)
            gate (defulat=0)
        has_skip_term: bool, optional
            whether to include a residual connection for output hidden
            state `h` (default=True)
        layer_norm: bool, optional
            whether to apply pre- layer normalization for this layer
            (default=False)
        rescale: bool, optional
            whether to apply a constant rescaling multiplier for the
            residual term (default=True)
        v1: bool, optional
            [DEPRECATED] whether to use the an ealier v1 implementation
            of SRU (default=False)
        transform_module: nn.Module, optional
            use the give module instead of the batched matrix
            multiplication to compute the intermediate representations U
            needed for the elementwise recurrrence operation
            (default=None)
        amp_recurrence_fp16: Type, optional
            When using AMP autocast, selects which type to use
            for recurrence custom kernel.
            False: torch.float32, True: torch.float16
        normalize_after: bool, optional
            if True use post layer norm, else pre layer norm
            (default=True)
        weight_c_init: float, optional
            size of uniform initiatialization of weight_c
            (default=1.0)
        """
        super(SRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size  # hidden size per direction
        self.output_size = hidden_size * 2 if bidirectional else hidden_size
        self.rnn_dropout = float(rnn_dropout)
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.has_skip_term = has_skip_term
        self.highway_bias = highway_bias
        self.v1 = v1
        self.rescale = rescale
        self.activation_type = 0
        self.activation = 'none'
        if use_tanh:
            self.activation_type = 1
            self.activation = 'tanh'
        self.amp_recurrence_fp16 = amp_recurrence_fp16
        self.normalize_after = normalize_after
        self.weight_c_init = weight_c_init

        # projection dimension
        self.projection_size = projection_size

        # number of sub-matrices used in SRU
        self.num_matrices = 3
        if has_skip_term and self.input_size != self.output_size:
            self.num_matrices = 4

        if transform_module is None:
            # create an appropriate transform_module, depending on whether we are using projection
            # or not
            if self.projection_size == 0:
                # use an nn.Linear
                transform_module = nn.Linear(input_size, self.output_size * self.num_matrices, bias=False)
            else:
                # use a Sequential[nn.Linear, nn.Linear]
                transform_module = nn.Sequential(nn.Linear(input_size, self.projection_size, bias=False),
                                                 nn.Linear(self.projection_size, self.output_size * self.num_matrices, bias=False),
                                                )
        self.transform_module: nn.Module = transform_module

        self.weight_c = nn.Parameter(torch.Tensor(2 * self.output_size))
        self.bias = nn.Parameter(torch.Tensor(2 * self.output_size))

        # scaling constant used in highway connections when rescale=True
        self.register_buffer('scale_x', torch.FloatTensor([0]))

        self.layer_norm: Optional[nn.Module] = None
        if layer_norm:
            self.layer_norm = nn.LayerNorm(self.output_size) if normalize_after else nn.LayerNorm(self.input_size)

        self.reset_parameters()
        SRUCell.init_elementwise_recurrence_funcs()

    @classmethod
    def init_elementwise_recurrence_funcs(cls):
        """
        Initializes the elementwise recurrence functions. This is postponed to the creation
        of the first SRUCell instance because we want to avoid eager CUDA initialization and
        ensure it takes place in the process running the model.
        """
        if not cls.initialized:        
            from ..ops import (elementwise_recurrence_inference,
                                elementwise_recurrence_gpu,
                                elementwise_recurrence_naive)
            cls.elementwise_recurrence_inference = elementwise_recurrence_inference
            cls.elementwise_recurrence_gpu = elementwise_recurrence_gpu
            cls.elementwise_recurrence_naive = elementwise_recurrence_naive
            cls.initialized = True


    def reset_parameters(self):
        """Initialize the weights of SRU.
        """
        # initialize bias and scaling constant
        self.bias.data.zero_()
        bias_val, output_size = self.highway_bias, self.output_size
        self.bias.data[output_size:].zero_().add_(bias_val)
        self.scale_x.data[0] = 1
        if self.rescale and self.has_skip_term:
            # scalar used to properly scale the highway output
            scale_val = (1 + math.exp(bias_val) * 2)**0.5
            self.scale_x.data[0] = scale_val

        def reset_module_parameters(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif hasattr(module, 'reset_parameters'):
                module.reset_parameters()  # type: ignore
            elif isinstance(module, nn.Sequential):
                for m in module:
                    reset_module_parameters(m)
            else:
                warnings.warn("Unable to reset parameters for custom module. "
                              "reset_parameters() method not found for custom module. "
                              + module.__class__.__name__)

        reset_module_parameters(self.transform_module)

        if not self.v1:
            self.weight_c.data.uniform_(-self.weight_c_init, self.weight_c_init)
        else:
            self.weight_c.data.zero_()
            self.weight_c.requires_grad = False

    def forward(self,
                input: Tensor,
                c0: Optional[Tensor] = None,
                mask_pad: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """The forward method of the SRU layer.
        """

        if input.dim() != 2 and input.dim() != 3:
            raise ValueError("Input must be 2 or 3 dimensional")

        batch_size = input.size(-2)
        if c0 is None:
            c0 = torch.zeros(batch_size, self.output_size, dtype=input.dtype,
                             device=input.device)

        # apply layer norm before activation (i.e. before SRU computation)
        residual = input

        layer_norm = self.layer_norm
        if layer_norm is not None:
            if not self.normalize_after:
                input = layer_norm(input)

        # apply dropout for multiplication
        if self.training and (self.rnn_dropout > 0):
            mask = self.get_dropout_mask_((batch_size, input.size(-1)), self.rnn_dropout)
            input = input * mask.expand_as(input)

        # get the scaling constant; scale_x is a scalar
        scale_val: Optional[Tensor] = None
        scale_val = self.scale_x if self.rescale else None

        # get dropout mask
        mask_c: Optional[Tensor] = None
        if self.training and (self.dropout > 0):
            mask_c = self.get_dropout_mask_((batch_size, self.output_size),
                                            self.dropout)

        # compute U, V
        #   U is (length, batch_size, output_size * num_matrices)
        #   V is (output_size*2,) or (length, batch_size, output_size * 2) if provided
        U, V = self.compute_UV(input, c0, mask_pad)

        # apply elementwise recurrence to get hidden states h and c
        h, c = self.apply_recurrence(U, V, residual, c0, scale_val, mask_c, mask_pad)

        if layer_norm is not None:
            if self.normalize_after:
                h = layer_norm(h)

        return h, c

    def apply_recurrence(self,
                         U: Tensor,
                         V: Tensor,
                         residual: Tensor,
                         c0: Tensor,
                         scale_val: Optional[Tensor],
                         mask_c: Optional[Tensor],
                         mask_pad: Optional[Tensor]) -> List[Tensor]:
        """
        Apply the elementwise recurrence computation on given input
        tensors

        """
        if not torch.jit.is_scripting():
            if self.bias.is_cuda:
                return SRUCell.elementwise_recurrence_gpu(U, residual, V, self.bias, c0,
                                                            self.activation_type,
                                                            self.hidden_size,
                                                            self.bidirectional,
                                                            self.has_skip_term,
                                                            scale_val, mask_c, mask_pad,
                                                            self.amp_recurrence_fp16)
            else:
                return SRUCell.elementwise_recurrence_naive(U, residual, V, self.bias, c0,
                                                            self.activation_type,
                                                            self.hidden_size,
                                                            self.bidirectional,
                                                            self.has_skip_term,
                                                            scale_val, mask_c, mask_pad)
        else:
            return SRUCell.elementwise_recurrence_inference(U, residual, V, self.bias, c0,
                                                            self.activation_type,
                                                            self.hidden_size,
                                                            self.bidirectional,
                                                            self.has_skip_term,
                                                            scale_val, mask_c, mask_pad)

    def compute_UV(self,
                   input: Tensor,
                   c0: Optional[Tensor],
                   mask_pad: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        SRU performs grouped matrix multiplication to transform the
        input (length, batch_size, input_size) into a tensor U of size
        (length * batch_size, output_size * num_matrices).

        U will be computed by the given transform_module. 
        The module can optionally return an additional 
        tensor V (length, batch_size, output_size * 2) 
        that will be added to the hidden-to-hidden coefficient terms in
        sigmoid gates, i.e., (V[t, b, d] + weight_c[d]) * c[t-1].
        """
        ret = self.transform_module(input)
        if isinstance(ret, tuple) or isinstance(ret, list):
            if len(ret) > 2:
                raise Exception(f"Custom module must return 1 or 2 tensors but got {len(ret)}.")
            U, V = ret[0], ret[1] + self.weight_c
        else:
            U, V = ret, self.weight_c

        if U.size(-1) != self.output_size * self.num_matrices:
            raise ValueError(f"U must have a last dimension of {self.output_size * self.num_matrices} but got {U.size(-1)}.")
        if V.size(-1) != self.output_size * 2:
            raise ValueError(f"V must have a last dimension of {self.output_size * 2} but got {V.size(-1)}.")

        return U, V

    def get_dropout_mask_(self,
                          size: Tuple[int, int],
                          p: float) -> Tensor:
        """
        Composes the dropout mask for the `SRUCell`.
        """
        b = self.bias.data
        return b.new_empty(size).bernoulli_(1 - p).div_(1 - p)

    def extra_repr(self):
        s = "{input_size}, {hidden_size}"
        if self.projection_size > 0:
            s += ", projection_size={projection_size}"
        if self.dropout > 0:
            s += ", dropout={dropout}"
        if self.rnn_dropout > 0:
            s += ", rnn_dropout={rnn_dropout}"
        if self.bidirectional:
            s += ", bidirectional={bidirectional}"
        if self.highway_bias != 0:
            s += ", highway_bias={highway_bias}"
        if self.activation_type != 0:
            s += ", activation={activation}"
        if self.v1:
            s += ", v1={v1}"
        if self.rescale:
            s += ", rescale={rescale}"
        if not self.has_skip_term:
            s += ", has_skip_term={has_skip_term}"
        if self.layer_norm:
            s += ", layer_norm=True"
            s += ", normalize_after={normalize_after}"
        s += ",\n  transform_module=" + str(self.transform_module)
        return s.format(**self.__dict__)

    def __repr__(self):
        s = self.extra_repr()
        if len(s.split('\n')) == 1:
            return "{}({})".format(self.__class__.__name__, s)
        else:
            return "{}({}\n)".format(self.__class__.__name__, s)


class SRU(nn.Module):
    """
    Implementation of Simple Recurrent Unit (SRU)
    """

    __constants__ = ['input_size', 'hidden_size', 'output_size', 'num_layers',
                     'dropout', 'rnn_dropout', 'projection_size', 'rnn_lst',
                     'bidirectional', 'use_layer_norm', 'has_skip_term',
                     'num_directions', 'nn_rnn_compatible_return', 'input_to_hidden']

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 2,
                 dropout: float = 0.0,
                 rnn_dropout: float = 0.0,
                 bidirectional: bool = False,
                 projection_size: Union[int, Sequence[int]] = 0,
                 highway_bias: float = 0.0,
                 layer_norm: bool = False,
                 normalize_after: bool = False,
                 transform_module: Optional[Union[nn.Module, Sequence[nn.Module]]] = None,
                 has_skip_term: bool = True,
                 rescale: bool = False,
                 use_tanh: bool = False,
                 v1: bool = False,
                 nn_rnn_compatible_return: bool = False,
                 proj_input_to_hidden_first: bool = False,
                 amp_recurrence_fp16: bool = True,
                 weight_c_init: float = 1.0):
        """Initialize the SRU module.

        Parameters
        ----------
        input_size: int
            the number of features in the input `x`
        hidden_size: int
            the number of features in the hidden state *for each
            direction*
        num_layers: int
            the number of stacked SRU layers (default=2)
        dropout: float, optional
            the dropout value applied between layers (default=0)
        rnn_dropout: float, optional
            [DEPRECATED] the variational dropout value (default=0)
            This option is deprecated because minimal performance
            improvement, and increases codebase size. This option will
            be removed at the next major version upgrade
        bidirectional: bool, optional
            if True, set the module as a bidirectional SRU
            (default=False)
        projection_size: Union[int, Sequence[int]]
            if non-zero, factorize the ``weight`` parameter in each
            layeras a product of two parameter matrices, using an inner
            dimension ``projection_size`` (default=0)
            If a sequence, length must equal number of layers, and
            values are projection size for each layer
        use_tanh: bool, optional
            [DEPRECATED] if True, apply `tanh` activation to the hidden
            state (default=False). `tanh` is deprecated because minimal
            performance improvement, and increases codebase size. This
            option will be removed at the next major version upgrade.
        layer_norm: bool, optional
            whether to apply pre- layer normalization for this layer
            (default=False)
        highway_bias: float, optional
            the initial value of the bias used in the highway (sigmoid)
            gate (defulat=0)
        has_skip_term: bool, optional
            whether to include a residual connection for output hidden
            state `h` (default=True)
        rescale: bool, optional
            whether to apply a constant rescaling multiplier for the
            residual term (default=False)
        v1: bool, optional
            [DEPRECATED] whether to use the an ealier v1 implementation
            of SRU (default=False)
        transform_module: Union[nn.Module, Sequence[nn.Module]], optional
            use the given module(s) instead of the batched matrix
            multiplication to compute the intermediate representations U
            needed for the elementwise recurrrence operation.  The
            module must take input x of shape (seq_len, batch_size,
            hidden_size). It returns a tensor U of shape (seq_len,
            batch_size, hidden_size * num_matrices), and one optional
            tensor V of shape (seq_len, batch_size, hidden_size * 2).
            (default=None)
        amp_recurrence_fp16: Type, optional
            When using AMP autocast, selects which type to use
            for recurrence custom kernel.
            False: torch.float32, True: torch.float16
        normalize_after: bool
            if True use post layer norm, else use pre layer norm
        weight_c_init: float, optional
            if not None, then size of uniform initiatialization of weight_c
            (default 1.0)
        """

        super(SRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = hidden_size * 2 if bidirectional else hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout
        self.projection_size = projection_size
        self.bidirectional = bidirectional
        self.use_layer_norm = layer_norm
        self.has_skip_term = has_skip_term
        self.num_directions = 2 if bidirectional else 1
        self.nn_rnn_compatible_return = nn_rnn_compatible_return
        self.input_to_hidden = None
        if proj_input_to_hidden_first and input_size != self.output_size:
            first_layer_input_size = self.output_size
            self.input_to_hidden = nn.Linear(input_size, self.output_size, bias=False)
        else:
            first_layer_input_size = input_size
        self.amp_recurrence_fp16 = amp_recurrence_fp16

        if rnn_dropout > 0:
            warnings.warn("rnn_dropout > 0 is deprecated and will be removed in"
                          "next major version of SRU. Please use dropout instead.")
        if use_tanh:
            warnings.warn("use_tanh = True is deprecated and will be removed in"
                          "next major version of SRU.")

        rnn_lst = nn.ModuleList()
        for i in range(num_layers):
            # get custom modules when provided
            transform_module_i = None
            if transform_module is not None:
                transform_module_i = transform_module[i] if isinstance(transform_module, list) else copy.deepcopy(transform_module)
            _projection_size = projection_size if isinstance(projection_size, int) else projection_size[i]
            # create the i-th SRU layer
            layer_i = SRUCell(
                first_layer_input_size if i == 0 else self.output_size,
                self.hidden_size,
                dropout=dropout if i + 1 != num_layers else 0,
                rnn_dropout=rnn_dropout,
                bidirectional=bidirectional,
                projection_size=_projection_size,
                use_tanh=use_tanh,
                layer_norm=layer_norm,
                highway_bias=highway_bias,
                has_skip_term=has_skip_term,
                rescale=rescale,
                v1=v1,
                transform_module=transform_module_i,
                amp_recurrence_fp16=amp_recurrence_fp16,
                normalize_after=normalize_after,
                weight_c_init=weight_c_init,
            )
            rnn_lst.append(layer_i)
        self.rnn_lst = rnn_lst

    def __getitem__(self, n: int) -> SRUCell:
        """
        returns n'th layer srucell
        """
        return self.rnn_lst[n]

    def forward(self, input: Tensor,
                c0: Optional[Tensor] = None,
                mask_pad: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """The forward method of SRU module

        Parameters
        ----------
        input: Tensor
            the input feature. shape: (length, batch_size, input_size)
        c0: Tensor, optional
            the initial internal hidden state. shape: (num_layers,
            batch_size, output_size) where
            output_size = hidden_size * num_direction
        mask_pad: Tensor, optional
            the mask where a non-zero value indicates if an input token
            is pad token that should be ignored in forward and backward
            computation. shape: (length, batch_size)

        Returns
        ----------
        h: Tensor
            the output hidden state. shape: (length, batch_size,
            output_size) where
            output_size = hidden_size * num_direction
        c: Tensor
            the last internal hidden state. shape: (num_layers,
            batch_size, output_size), or (num_layers * num_directions,
            batch_size, hidden_size) if `nn_rnn_compatible_return` is
            set `True`

        """
        # unpack packed, if input is packed. packing and then unpacking will be slower than not
        # packing at all, but makes SRU usage compatible with nn.RNN usage
        orig_input = input
        if isinstance(orig_input, PackedSequence):
            input, lengths = nn.utils.rnn.pad_packed_sequence(input)
            max_length = lengths.max().item()
            mask_pad = torch.ByteTensor([[0] * length + [1] * (max_length - length)
                                        for length in lengths.tolist()])
            mask_pad = mask_pad.to(input.device).transpose(0, 1).contiguous()

        # The dimensions of `input` should be: `(sequence_length, batch_size, input_size)`.
        if input.dim() != 3:
            raise ValueError("There must be 3 dimensions for (length, batch_size, input_size)")

        if c0 is None:
            zeros = torch.zeros(input.size(1), self.output_size, dtype=input.dtype,
                                device=input.device)
            c0_ = [zeros for i in range(self.num_layers)]
        else:
            # The dimensions of `c0` should be: `(num_layers, batch_size, hidden_size * dir_)`.
            if c0.dim() != 3:
                raise ValueError("c0 must be 3 dim (num_layers, batch_size, output_size)")
            c0_ = [x.squeeze(0) for x in c0.chunk(self.num_layers, 0)]

        if self.input_to_hidden is None:
            prevx = input
        else:
            prevx = self.input_to_hidden(input)
        lstc = []
        i = 0
        for rnn in self.rnn_lst:
            h, c = rnn(prevx, c0_[i], mask_pad=mask_pad)
            prevx = h
            lstc.append(c)
            i += 1

        lstc_stack = torch.stack(lstc)
        if self.nn_rnn_compatible_return:
            batch_size = input.size(1)
            lstc_stack = lstc_stack.view(self.num_layers, batch_size,
                                         self.num_directions, self.hidden_size)
            lstc_stack = lstc_stack.transpose(1, 2).contiguous()
            lstc_stack = lstc_stack.view(self.num_layers * self.num_directions,
                                         batch_size, self.hidden_size)

        if isinstance(orig_input, PackedSequence):
            prevx = nn.utils.rnn.pack_padded_sequence(prevx, lengths, enforce_sorted=False)
            return prevx, lstc_stack
        else:
            return prevx, lstc_stack

    def reset_parameters(self):
        for rnn in self.rnn_lst:
            rnn.reset_parameters()
        if self.input_to_hidden is not None:
            self.input_to_hidden.reset_parameters()