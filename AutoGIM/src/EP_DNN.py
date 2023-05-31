import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block


class EP_DNN(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="EP_DNN",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 hidden_units=[64, 64, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(EP_DNN, self).__init__(feature_map,
                                     model_id=model_id,
                                     gpu=gpu,
                                     embedding_regularizer=embedding_regularizer,
                                     net_regularizer=net_regularizer,
                                     **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.num_fields = feature_map.num_fields
        self.input_dim = int(self.num_fields * (self.num_fields + 1) / 2) * embedding_dim
        self.triu_node_index = nn.Parameter(torch.triu_indices(self.num_fields, self.num_fields, offset=0),
                                            requires_grad=False)
        self.mlp = MLP_Block(input_dim=self.input_dim,
                             output_dim=1,
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=self.output_activation,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        flattened_emb = self.ep_flatten_emb(feature_emb)
        y_pred = self.mlp(flattened_emb)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def ep_flatten_emb(self, h):
        emb1 = torch.index_select(h, 1, self.triu_node_index[0])
        emb2 = torch.index_select(h, 1, self.triu_node_index[1])
        embs = emb1 * emb2
        flattened_emb = embs.view(-1, self.input_dim)
        return flattened_emb
