import torch
from torch import nn
from torch.functional import F
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block


class AutoGIM_NFNAS(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="AutoGIM_NFNAS",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 fi_hidden_units=[64, 64, 64],
                 w_hidden_units=[64, 64, 64],
                 hidden_activations="leaky_relu",
                 warm_dim=10,
                 cold_dim=10,
                 warm_tau=1.0,
                 cold_tau=0.01,
                 only_use_last_layer=True,
                 continuous_patience=10,
                 gnn_layers=3,
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(AutoGIM_NFNAS, self).__init__(feature_map,
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
        self.warm_dim = warm_dim
        self.cold_dim = cold_dim
        self.warm_tau = warm_tau
        self.cold_tau = cold_tau
        self.only_use_last_layer = only_use_last_layer
        self.gnn_layers = gnn_layers
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
        self.continuous_patience = continuous_patience
        self.W1_mistake = 0
        self.W2_mistake = 0
        self.W3_mistake = 0

        self.AutoGraph = AutoGraph_Layer(num_fields=self.num_fields,
                                         embedding_dim=self.embedding_dim,
                                         warm_dim=self.warm_dim,
                                         cold_dim=self.cold_dim,
                                         warm_tau=self.warm_tau,
                                         cold_tau=self.cold_tau,
                                         only_use_last_layer=self.only_use_last_layer,
                                         gnn_layers=self.gnn_layers)
        final_score_weight = torch.rand(self.gnn_layers)
        final_score_weight = final_score_weight / torch.sum(final_score_weight)
        self.final_score_weight = nn.Parameter(final_score_weight, requires_grad=True)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        """
        Inputs: [X,y]
        """
        X = self.get_inputs(inputs)
        y_pred = 0
        feature_emb = self.embedding_layer(X)
        h_list = self.AutoGraph(feature_emb)
        for h, fsw in zip(h_list, self.final_score_weight):
            embs_list = self.ep_ip_flatten_emb(h)
            WX_list = self.select_interaction_model(embs_list)
            for WX in WX_list:
                y_pred += WX
            if not self.only_use_last_layer:
                y_pred = y_pred * fsw
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


class AutoGraph_Layer(nn.Module):
    def __init__(self,
                 num_fields,
                 embedding_dim,
                 warm_dim,
                 cold_dim,
                 warm_tau=1.0,
                 cold_tau=0.01,
                 only_use_last_layer=True,
                 gnn_layers=3):
        super(AutoGraph_Layer, self).__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.gnn_layers = gnn_layers
        self.only_use_last_layer = only_use_last_layer
        self.warm_dim = warm_dim
        self.cold_dim = cold_dim
        self.warm_tau = warm_tau
        self.cold_tau = cold_tau
        self.all_node_index = nn.Parameter(
            torch.triu_indices(self.num_fields, self.num_fields, offset=-(self.num_fields - 1)), requires_grad=False)
        self.triu_node_index = nn.Parameter(torch.triu_indices(self.num_fields, self.num_fields, offset=0),
                                            requires_grad=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.warm_W_transform = nn.Sequential(nn.Linear(self.embedding_dim, self.warm_dim),
                                              nn.ReLU(),
                                              nn.Linear(self.warm_dim, 1, bias=False))
        self.cold_W_transform = nn.Linear(self.embedding_dim * 2, 1, bias=False)
        self.W_GraphSage = torch.nn.Parameter(torch.Tensor(self.num_fields, self.embedding_dim, self.embedding_dim),
                                              requires_grad=True)
        nn.init.xavier_normal_(self.W_GraphSage)
        self.eye_mask = nn.Parameter(torch.eye(self.num_fields).bool(), requires_grad=False)

    def build_warm_matrix(self, feature_emb, warm_tau):
        transformer = self.warm_W_transform(feature_emb)
        warm_matrix = F.gumbel_softmax(transformer, tau=warm_tau, dim=1)
        return warm_matrix

    def build_cold_matrix(self, feature_emb, cold_tau):  # Sparse adjacency matrix
        emb1 = torch.index_select(feature_emb, 1, self.all_node_index[0])
        emb2 = torch.index_select(feature_emb, 1, self.all_node_index[1])
        concat_emb = torch.cat((emb1, emb2), dim=-1)
        alpha = self.leaky_relu(self.cold_W_transform(concat_emb))
        alpha = alpha.view(-1, self.num_fields, self.num_fields)
        cold_matrix = F.gumbel_softmax(alpha, tau=cold_tau, dim=-1)
        mask = cold_matrix.gt(0)
        cold_matrix_allone = torch.masked_fill(cold_matrix, mask, float(1))
        cold_matrix_allone = torch.masked_fill(cold_matrix_allone, self.eye_mask, float(1))
        return cold_matrix_allone

    def forward(self, feature_emb):
        h = feature_emb
        h_list = []
        for i in range(self.gnn_layers):
            cold_matrix = self.build_cold_matrix(h, self.cold_tau)
            new_feature_emb = torch.bmm(cold_matrix, h)
            new_feature_emb = torch.matmul(self.W_GraphSage, new_feature_emb.unsqueeze(-1)).squeeze(-1)
            warm_matrix = self.build_warm_matrix(new_feature_emb, self.warm_tau)
            new_feature_emb = new_feature_emb * warm_matrix
            new_feature_emb = self.leaky_relu(new_feature_emb)
            if not self.only_use_last_layer:
                h_list.append(h)
            else:
                if self.gnn_layers == (i + 1):
                    h_list.append(h)
            h = new_feature_emb + feature_emb  # ResNet
        return h_list
