import torch
import torch.nn.functional as F
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block


class MFIM(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="MFIM",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 fi_hidden_units=[64, 64, 64],
                 w_hidden_units=[64, 64, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(MFIM, self).__init__(feature_map,
                                  model_id=model_id,
                                  gpu=gpu,
                                  embedding_regularizer=embedding_regularizer,
                                  net_regularizer=net_regularizer,
                                  **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        self.embedding_dim = embedding_dim
        self.num_fields = feature_map.num_fields
        self.ep_dim = int(self.num_fields * (self.num_fields + 1) / 2) * self.embedding_dim
        self.ip_dim = int(self.num_fields * (self.num_fields + 1) / 2)
        self.flatten_dim = feature_map.sum_emb_out_dim()
        self.triu_node_index = nn.Parameter(torch.triu_indices(self.num_fields, self.num_fields, offset=0),
                                            requires_grad=False)
        self.mlp1 = MLP_Block(input_dim=self.ep_dim,
                              output_dim=1,
                              hidden_units=fi_hidden_units,
                              hidden_activations=hidden_activations,
                              output_activation=None,
                              dropout_rates=net_dropout,
                              batch_norm=batch_norm)
        self.W1 = MLP_Block(input_dim=self.ep_dim,
                            output_dim=1,
                            hidden_units=w_hidden_units,
                            hidden_activations=hidden_activations,
                            output_activation='leaky_relu',
                            dropout_rates=net_dropout,
                            batch_norm=batch_norm)
        self.mlp2 = MLP_Block(input_dim=self.ip_dim,
                              output_dim=1,
                              hidden_units=fi_hidden_units,
                              hidden_activations=hidden_activations,
                              output_activation=None,
                              dropout_rates=net_dropout,
                              batch_norm=batch_norm)
        self.W2 = MLP_Block(input_dim=self.ip_dim,
                            output_dim=1,
                            hidden_units=w_hidden_units,
                            hidden_activations=hidden_activations,
                            output_activation='leaky_relu',
                            dropout_rates=net_dropout,
                            batch_norm=batch_norm)
        self.mlp3 = MLP_Block(input_dim=self.flatten_dim,
                              output_dim=1,
                              hidden_units=fi_hidden_units,
                              hidden_activations=hidden_activations,
                              output_activation=None,
                              dropout_rates=net_dropout,
                              batch_norm=batch_norm)
        self.W3 = MLP_Block(input_dim=self.flatten_dim,
                            output_dim=1,
                            hidden_units=w_hidden_units,
                            hidden_activations=hidden_activations,
                            output_activation='leaky_relu',
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
        embs_ep, embs_ip, embs_flatten = self.ep_ip_flatten_emb(feature_emb)
        W1 = self.W1(embs_ep)
        X1 = self.mlp1(embs_ep)
        W2 = self.W2(embs_ip)
        X2 = self.mlp2(embs_ip)
        W3 = self.W3(embs_flatten)
        X3 = self.mlp3(embs_flatten)
        y_pred = W1*X1 + W2*X2 + W3*X3
        y_pred = self.output_activation(y_pred)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def ep_ip_flatten_emb(self, h):
        emb1 = torch.index_select(h, 1, self.triu_node_index[0])
        emb2 = torch.index_select(h, 1, self.triu_node_index[1])
        embs_ep = emb1 * emb2
        embs_ip = torch.sum(embs_ep, dim=-1)
        embs_ep = embs_ep.view(-1, self.ep_dim)
        embs_flatten = h.flatten(start_dim=1)
        return embs_ep, embs_ip, embs_flatten
