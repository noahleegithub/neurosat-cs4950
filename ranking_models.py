from torch import nn, Tensor
import yaml
import json
from types import SimpleNamespace
from models import MultiLayerPerceptron

class DirectRanker(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int, p_drop=0.0, xavier_init=False, leaky_slope=0.01):
        super().__init__()
        self.enc_ffnn = MultiLayerPerceptron(layer_dims=[in_dim]+[hidden_dim]*n_layers,
            layer_activations=[nn.LeakyReLU(negative_slope=leaky_slope)]*n_layers, 
            p_dropout=p_drop, xavier_init=xavier_init)
        self.out_ffnn = MultiLayerPerceptron(layer_dims=[hidden_dim, 1], layer_activations=[nn.Identity()],
            p_dropout=p_drop, bias=False, xavier_init=xavier_init)

    def forward(self, x: Tensor):
        N = x.shape[-2]
        assert N == 2
        x1 = x[..., 0, :]
        x2 = x[..., 1, :]

        x1 = self.enc_ffnn(x1)
        x2 = self.enc_ffnn(x2)

        return self.out_ffnn(x1 - x2)

if __name__ == "__main__":
    with open("configurations/config_rank.yaml") as yaml_reader:
        config = yaml.safe_load(yaml_reader)
        config = json.loads(json.dumps(config), object_hook=lambda d : SimpleNamespace(**d))
    model = DirectRanker(config.rank_model.input_dim, config.rank_model.mlp_hidden_dim, config.rank_model.mlp_n_layers, 
        p_drop=config.rank_model.mlp_p_dropout, xavier_init=config.rank_model.xavier_init, 
        leaky_slope=config.rank_model.leaky_slope)
    print(model)