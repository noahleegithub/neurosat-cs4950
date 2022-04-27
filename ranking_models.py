from torch import nn, Tensor
import yaml
import json
from types import SimpleNamespace
from models import MultiLayerPerceptron

class DirectRanker(nn.Module):

    def __init__(self, config):
        super().__init__()
        in_dim = config.rank_model.input_dim
        hidden_dim = config.rank_model.mlp_hidden_dim
        n_layers = config.rank_model.mlp_n_layers
        p_drop = config.rank_model.mlp_p_dropout
        xavier = config.rank_model.xavier_init
        self.doc_ffnn = MultiLayerPerceptron(layer_dims=[in_dim]+[hidden_dim]*n_layers,
            layer_activations=[nn.ReLU()]*n_layers, p_dropout=p_drop, xavier_init=xavier)
        self.out_ffnn = MultiLayerPerceptron(layer_dims=[hidden_dim, 1], layer_activations=[nn.Identity()],
            p_dropout=p_drop, bias=False, xavier_init=xavier)

    def forward(self, docs: Tensor):
        B, N, D = docs.size()
        assert N == 2
        doc1 = docs[:, 0]
        doc2 = docs[:, 1]

        doc1 = self.doc_ffnn(doc1)
        doc2 = self.doc_ffnn(doc2)

        doc_diffs = doc1 - doc2

        out = self.out_ffnn(doc_diffs)

        return out

# a good loss would be min(x, 0)**2

if __name__ == "__main__":
    with open("configurations/config_rank.yaml") as yaml_reader:
        config = yaml.safe_load(yaml_reader)
        config = json.loads(json.dumps(config), object_hook=lambda d : SimpleNamespace(**d))
    model = DirectRanker(config)
    print(model)