# =========================================================================
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block


class MFIM_NFNAS(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="MFIM_NFNAS",
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
        super(MFIM_NFNAS, self).__init__(feature_map,
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
        self.continuous_patience = 10
        self.W1_mistake = 0
        self.W2_mistake = 0
        self.W3_mistake = 0

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)
        embs_list = self.ep_ip_flatten_emb(feature_emb)
        y_pred = 0
        WX_list = self.select_interaction_model(embs_list)
        for WX in WX_list:
            y_pred = y_pred + WX
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
        return [embs_ep, embs_ip, embs_flatten]

    def select_interaction_model(self, embs_list):
        if self.W1_mistake <= self.continuous_patience:
            W1 = self.W1(embs_list[0])
            if torch.all(W1 <= 0):
                self.W1_mistake = self.W1_mistake + 1
            else:
                self.W1_mistake = 0
            X1 = self.mlp1(embs_list[0])
            W1X1 = W1 * X1
        else:
            W1X1 = 0

        if self.W2_mistake <= self.continuous_patience:
            W2 = self.W2(embs_list[1])
            if torch.all(W2 <= 0):
                self.W2_mistake = self.W2_mistake + 1
            else:
                self.W2_mistake = 0
            X2 = self.mlp2(embs_list[1])
            W2X2 = W2 * X2
        else:
            W2X2 = 0

        if self.W3_mistake <= self.continuous_patience:
            W3 = self.W3(embs_list[2])
            if torch.all(W3 <= 0):
                self.W3_mistake = self.W3_mistake + 1
            else:
                self.W3_mistake = 0
            X3 = self.mlp3(embs_list[2])
            W3X3 = W3 * X3
        else:
            W3X3 = 0
        return [W1X1, W2X2, W3X3]
