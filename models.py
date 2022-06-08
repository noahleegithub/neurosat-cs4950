import math
from types import SimpleNamespace
from typing import List, Tuple

import torch
from torch import Tensor
from torch import nn
from torch.nn import Module, Parameter

def activations(config: SimpleNamespace, activation: str):
    activations_dict = {
        'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh(), 'softmax': nn.Softmax(),
        'identity': nn.Identity(), 'leakyrelu': nn.LeakyReLU(config.activations.leaky_slope),
    }
    return activations_dict[activation]

class MultiLayerPerceptron(Module):

    def __init__(self, layer_dims: List[int], layer_activations: List[Module], p_dropout=0.0, bias=True,
            xavier_init=False) -> None:
        super().__init__()
        if not len(layer_dims) - 1 == len(layer_activations):
            raise ValueError("len(layer_dims) - 1 must equal len(layer_activations)")
        self.xavier_init = xavier_init
        
        modules = []
        for i in range(len(layer_dims) - 1):
            modules.append(nn.Linear(layer_dims[i], layer_dims[i + 1], bias=bias))
            if i != len(layer_dims) - 2:
                modules.append(nn.Dropout(p=p_dropout))
            modules.append(layer_activations[i])

        self.layers = nn.Sequential(*modules)
        self.layers.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            if self.xavier_init:
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.01)

    def forward(self, input: Tensor) -> Tensor:
        return self.layers(input)

class LayerNormLSTMCell(Module):

    def __init__(self, input_size:int, hidden_size: int, activation: Module=nn.Tanh(), p_dropout=0.0, forget_bias=0.0) -> None:
        super().__init__()
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.activation = activation

        self.layernorm_i = nn.LayerNorm(4 * hidden_size)
        self.layernorm_h = nn.LayerNorm(4 * hidden_size)
        self.layernorm_c = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=p_dropout)

        with torch.no_grad():
            self.layernorm_i.bias[hidden_size : 2 * hidden_size].fill_(forget_bias)
            self.layernorm_h.bias[hidden_size : 2 * hidden_size].fill_(forget_bias)

    def forward(self, input:Tensor, state:Tuple[Tensor,Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        igates = self.layernorm_i(torch.matmul(input, self.weight_ih.t()))
        hgates = self.layernorm_h(torch.matmul(hx, self.weight_hh.t()))
        gates = igates + hgates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, dim=-1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = self.layernorm_c((forgetgate * cx) + (ingate * cellgate))
        hy = outgate * self.activation(cy)
        hy = self.dropout(hy)

        return hy, (hy, cy)

class LayerNormGRUCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, reset_bias=0.0):
        super(LayerNormGRUCell, self).__init__()

        self.ln_i2h = torch.nn.LayerNorm(2*hidden_size, elementwise_affine=False)
        self.ln_h2h = torch.nn.LayerNorm(2*hidden_size, elementwise_affine=False)
        self.ln_cell_1 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln_cell_2 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.i2h = torch.nn.Linear(input_size, 2 * hidden_size, bias=bias)
        self.h2h = torch.nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        self.h_hat_W = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.h_hat_U = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.hidden_size = hidden_size
        self.reset_parameters()
        with torch.no_grad():
            self.i2h.bias[-hidden_size:].fill_(reset_bias)
            self.h2h.bias[-hidden_size:].fill_(reset_bias)


    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, h):

        h = h
        h = h.view(h.size(0), -1)
        x = x.view(x.size(0), -1)

        # Linear mappings
        i2h = self.i2h(x)
        h2h = self.h2h(h)

        # Layer norm
        i2h = self.ln_i2h(i2h)
        h2h = self.ln_h2h(h2h)

        preact = i2h + h2h

        # activations
        gates = preact[:, :].sigmoid()
        z_t = gates[:, :self.hidden_size]
        r_t = gates[:, -self.hidden_size:]

        # h_hat
        h_hat_first_half = self.h_hat_W(x)
        h_hat_last_half = self.h_hat_U(h)

        # layer norm
        h_hat_first_half = self.ln_cell_1( h_hat_first_half )
        h_hat_last_half = self.ln_cell_2( h_hat_last_half )

        h_hat = torch.tanh(  h_hat_first_half + torch.mul(r_t,   h_hat_last_half ) )

        h_t = torch.mul( 1-z_t , h ) + torch.mul( z_t, h_hat)

        # Reshape for compatibility

        h_t = h_t.view( h_t.size(0), -1)
        return h_t

if __name__ == "__main__":
    mlp_model = MultiLayerPerceptron([64, 128, 128, 64], [nn.ReLU()] * 3, p_dropout=0.1, bias=True,
        xavier_init=False)
    print(mlp_model)

    layernorm_lstm = LayerNormLSTMCell(64, 128)
    print(layernorm_lstm)

    B, I = 10, 64

    mlp_out = mlp_model(torch.rand((B,I)))
    print(mlp_out)

    lstm_out = layernorm_lstm(torch.rand((B,I)), (torch.rand((B,I*2)), torch.rand((B,I*2))))
    print(lstm_out)

