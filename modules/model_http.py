from cProfile import label
from cmath import log
from operator import neg
import random
from re import T
from turtle import forward
from typing import final
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.ops as ops
import dgl.function as fn
from torch_scatter import scatter_mean


class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network
    """
    def __init__(self, n_users, n_items, n_intents, emb_size,
                 c_index, v_index, k_index, other_index, mess_dropout_rate):
        super(Aggregator, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_intents = n_intents
        self.c_index = c_index
        self.v_index = v_index
        self.k_index = k_index
        self.other_index = other_index
        
        self.dropout = nn.Dropout(mess_dropout_rate)
        self.gru = nn.GRU(emb_size, emb_size, batch_first=True)

    def forward(self, entity_emb, user_emb, intent_emb,
                edge_index, edge_type, interact_mat, r_emb):

        n_entities = entity_emb.shape[0]
        emb_size = entity_emb.shape[1]
        n_users = self.n_users
        n_intents = self.n_intents

        """side infomation layer aggregate"""
        head, tail = edge_index
        edge_relation_emb = r_emb[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, emb_size]
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)

        """user intent"""
        c_node = entity_emb[-5, :].unsqueeze(0)  # course virtual node
        v_node = entity_emb[-4, :].unsqueeze(0)  # video virtual node
        k_node = entity_emb[-3, :].unsqueeze(0)  # knowledge virtual node
        t_node = entity_emb[-2, :].unsqueeze(0)  # teacher virtual node
        s_node = entity_emb[-1, :].unsqueeze(0)  # school virtual node

        o_emb = torch.cat([r_emb], dim=0)  # all relations and virtual nodes
        overall_intent_att = nn.Softmax()((intent_emb[0].expand(len(o_emb), emb_size) * o_emb).sum(dim=1)).unsqueeze(1)
        overall_intent = (overall_intent_att * o_emb).mean(dim=0).unsqueeze(dim=0)

        c_r_emb = torch.index_select(r_emb, 0, self.c_index)
        c_emb = torch.cat([c_r_emb], dim=0)
        c_intent_att = nn.Softmax()((intent_emb[1].expand(len(c_emb), emb_size) * c_emb).sum(dim=1)).unsqueeze(1)
        c_intent = (c_intent_att * c_emb).mean(dim=0).unsqueeze(dim=0)

        v_r_emb = torch.index_select(r_emb, 0, self.v_index)
        v_emb = torch.cat([v_r_emb], dim=0)
        v_intent_att = nn.Softmax()((intent_emb[2].expand(len(v_emb), emb_size) * v_emb).sum(dim=1)).unsqueeze(1)
        v_intent = (v_intent_att * v_emb).mean(dim=0).unsqueeze(dim=0)

        k_r_emb = torch.index_select(r_emb, 0, self.k_index)
        k_emb = torch.cat([k_r_emb], dim=0)
        k_intent_att = nn.Softmax()((intent_emb[3].expand(len(k_emb), emb_size) * k_emb).sum(dim=1)).unsqueeze(1)
        k_intent = (k_intent_att * k_emb).mean(dim=0).unsqueeze(dim=0)

        other_r_emb = torch.index_select(r_emb, 0, self.other_index)
        other_emb = torch.cat([other_r_emb], dim=0)
        other_intent_att = nn.Softmax()((intent_emb[4].expand(len(other_emb), emb_size) * other_emb).sum(dim=1)).unsqueeze(1)
        other_intent = (other_intent_att * other_emb).mean(dim=0).unsqueeze(dim=0)
        
        all_intent = torch.cat((overall_intent, c_intent, v_intent, k_intent, other_intent), axis=0)
        intent_emb = (all_intent + intent_emb) / 2

        user_intent = intent_emb.expand(n_users, n_intents, emb_size)

        """cul user-intent attention"""
        score_ = torch.mm(user_emb, intent_emb.t())
        score = nn.Softmax(dim=1)(score_).unsqueeze(-1)  # [n_users, n_intents, 1]

        """user aggregate"""
        user_agg = torch.sparse.mm(interact_mat, entity_emb)  # [n_users, emb_size]
        user_agg = user_agg * (user_intent * score).sum(dim=1) + user_agg  # [n_users, emb_size]

        return entity_agg, user_agg


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, n_hops, n_users, n_items, n_intents, ind, emb_size,
                 c_index, v_index, k_index, other_index,
                 node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.n_users = n_users
        self.n_items = n_items
        self.n_intents = n_intents
        self.emb_size = emb_size
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind

        self.temperature = 0.2

        self.c_index = c_index
        self.v_index = v_index
        self.k_index = k_index
        self.other_index = other_index

        for i in range(n_hops):
            self.convs.append(Aggregator(n_users=n_users,
                                         n_items=n_items,
                                         n_intents=n_intents,
                                         emb_size=self.emb_size,
                                         c_index=self.c_index,
                                         v_index=self.v_index,
                                         k_index=self.k_index,
                                         other_index=self.other_index,
                                         mess_dropout_rate=self.mess_dropout_rate))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()  # not zero elements

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    # def _cul_cor_pro(self):
    #     # disen_T: [num_factor, dimension]
    #     disen_T = self.factor_relation_att.t()
    #
    #     # normalized_disen_T: [num_factor, dimension]
    #     normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)
    #
    #     pos_scores = torch.sum(normalized_disen_T * normalized_disen_T, dim=1)
    #     ttl_scores = torch.sum(torch.mm(disen_T, self.factor_relation_att), dim=1)
    #
    #     pos_scores = torch.exp(pos_scores / self.temperature)
    #     ttl_scores = torch.exp(ttl_scores / self.temperature)
    #
    #     mi_score = - torch.sum(torch.log(pos_scores / ttl_scores))
    #     return mi_score

    def _cul_cor(self, intent_emb):
        def CosineSimilarity(tensor_1, tensor_2):
            # tensor_1, tensor_2: [emb_size]
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=0, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=0, keepdim=True)
            return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0) ** 2  # no negative
        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [emb_size]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            emb_size = tensor_1.shape[0]
            zeros = torch.zeros(emb_size, emb_size).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                   torch.matmul(tensor_2, tensor_2.t()) * 2  # [emb_size, emb_size]
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                   torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [emb_size, emb_size]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / emb_size ** 2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / emb_size ** 2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / emb_size ** 2, zero) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)
        def MutualInformation():
            # disen_T: [num_factor, dimension]
            disen_T = self.factor_relation_att.t()

            # normalized_disen_T: [num_factor, dimension]
            normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)

            pos_scores = torch.sum(normalized_disen_T * normalized_disen_T, dim=1)
            ttl_scores = torch.sum(torch.mm(disen_T, self.factor_relation_att), dim=1)

            pos_scores = torch.exp(pos_scores / self.temperature)
            ttl_scores = torch.exp(ttl_scores / self.temperature)

            mi_score = - torch.sum(torch.log(pos_scores / ttl_scores))
            return mi_score

        """cul similarity for each latent factor weight pairs"""
        if self.ind == 'mi':
            return MutualInformation()
        else:
            cor = 0
            for i in range(self.n_intents):
                for j in range(i + 1, self.n_intents):
                    if self.ind == 'distance':
                        cor += DistanceCorrelation(intent_emb[i], intent_emb[j])
                    else:
                        cor += CosineSimilarity(self.factor_relation_att[i], self.factor_relation_att[j])
        return cor

    def forward(self, user_emb, entity_emb, intent_emb, edge_index, edge_type, interact_mat,
                r_emb, mess_dropout=True, node_dropout=False):

        """node dropout"""
        if node_dropout:
            edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)
            interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)

        entity_res_emb = entity_emb  # [n_entity, emb_size]
        user_res_emb = user_emb  # [n_users, emb_size]
        cor = self._cul_cor(intent_emb)
        for i in range(len(self.convs)):
            entity_emb, user_emb = self.convs[i](entity_emb, user_emb, intent_emb,
                                                 edge_index, edge_type, interact_mat,
                                                 r_emb)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

        return entity_res_emb, user_res_emb, cor


class Readout(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        batch_norm=False,
        feat_drop=0.0,
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_u = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_v = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc_e = nn.Linear(hidden_dim, 1, bias=False)
        self.fc_out = (
            nn.Linear(input_dim, output_dim, bias=False)
            if output_dim != input_dim else None
        )
        self.activation = nn.PReLU(output_dim)

    def forward(self, g, feat, last_nodes):
        if self.batch_norm is not None:
            feat = self.batch_norm(feat)
        feat = self.feat_drop(feat) # [n_nodes_of_g, iput_size]
        feat_u = self.fc_u(feat) # [n_nodes_of_g, output_dim]
        feat_v = self.fc_v(feat[last_nodes]) # [batch_size, output_dim]

        # broadcast feat_v[i] to all nodes of g[i]
        feat_v = dgl.broadcast_nodes(g, feat_v) # [n_nodes_of_g, output_dim]
        e = self.fc_e(torch.sigmoid(feat_u + feat_v)) # [n_nodes_of_g, 1]

        # each graph in batch computes softmax independently
        alpha = ops.segment.segment_softmax(g.batch_num_nodes(), e) # [n_nodes_of_g, 1]
        feat_norm = feat * alpha

        ## aggregate each graph feat in batch
        rst = ops.segment.segment_reduce(g.batch_num_nodes(), feat_norm, 'sum') # [batch_size, input_dim]
        if self.fc_out is not None:
            rst = self.fc_out(rst)
        rst = self.activation(rst)
        return rst

class GRUCov(nn.Module):
    def __init__(self, input_dim, output_dim, feat_drop):
        super().__init__()
        # self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.gru = nn.GRU(input_dim, input_dim, batch_first=True)
        self.fc_self = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neigh = nn.Linear(input_dim, output_dim, bias=False)
        self.activation = nn.PReLU(output_dim)

    def reducer(self, nodes):
        m = nodes.mailbox['m']  # (num_nodes, deg, d)
        # m[i]: the messages passed to the i-th node with in-degree equal to 'deg'
        # the order of messages follows the order of incoming edges
        _, hn = self.gru(m)  # hn: (1, num_nodes, d)
        return {'neigh': hn.squeeze(0)}

    def forward(self, ug, feat, user, last_nodes):
        with ug.local_scope():
            ug.ndata['ft'] = self.feat_drop(feat)
            if ug.number_of_edges() > 0:
                try:
                    ug.update_all(fn.copy_u('ft', 'm'), self.reducer)
                    neigh = ug.ndata['neigh']
                    rst = self.fc_self(feat) + self.fc_neigh(neigh)
                except:
                    rst = self.fc_self(feat)
            else:
                rst = self.fc_self(feat)
            rst = self.activation(rst)
        return rst


class AttentionLayer(nn.Module):
    def __init__(self, att_hidden_size=[80, 40], dropout=0.1):
        super().__init__()

        fc_modules = []
        for (input_size, output_size) in zip(att_hidden_size[:-1], att_hidden_size[1:]):
            fc_modules.append(nn.Dropout(p=dropout))
            fc_modules.append(nn.Linear(input_size, output_size))
            # fc_modules.append(nn.BatchNorm1d(num_features=output_size))
            fc_modules.append(nn.LeakyReLU())
        
        fc_modules.append(nn.Linear(output_size, 1))
        fc_modules.append(nn.Sigmoid())

        self.fc_att = nn.Sequential(*fc_modules)

    def forward(self, queries, keys, last_node):
        emb_size = queries.shape[-1]
        keys_len = keys.shape[1]
        
        queries = queries.repeat(1, keys_len)
        queries = queries.view(-1, keys_len, emb_size)
        last_node = last_node.repeat(1, keys_len)
        last_node = last_node.view(-1, keys_len, emb_size * 4)

        input_tensor = torch.cat(
            [queries, keys, queries - keys, queries * keys, last_node], dim=-1
        )
        output = torch.transpose(self.fc_att(input_tensor), -1, -2)
        output = output / (emb_size ** 0.5)

        return torch.matmul(output, keys).squeeze(dim=1) # batch_size * emb_size
          

class Recommender(nn.Module):
    def __init__(self, data_config, args_config, graph, adj_mat):
        super(Recommender, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.decay = args_config.l2
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.n_intents = args_config.n_intents
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.ind = args_config.ind
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
                                                                      else torch.device("cpu")

        self.adj_mat = adj_mat
        self.graph = graph
        self.edge_index, self.edge_type = self._get_edges(graph)
        # [n_users, n_entities]
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

        # relation index about course, video and knowledge
        self.c_index = torch.tensor([1, 3]).to(self.device)
        self.v_index = torch.tensor([0, 1, 2]).to(self.device)
        self.k_index = torch.tensor([0]).to(self.device)
        self.other_index = torch.tensor([1, 2, 3, 4]).to(self.device)

        initializer = nn.init.xavier_uniform_
        self.all_emb = nn.Parameter(initializer(torch.empty(self.n_nodes, self.emb_size)))
        self.intent_emb = nn.Parameter(initializer(torch.empty(self.n_intents, self.emb_size)))

        # relation embedding, not include interact
        self.r_emb = nn.Parameter(initializer(torch.empty(self.n_relations - 1, self.emb_size)))

        self.gcn = GraphConv(n_hops=self.context_hops,
                             n_users=self.n_users,
                             n_items=self.n_items,
                             n_intents=self.n_intents,
                             ind=self.ind,
                             emb_size=self.emb_size,
                             c_index=self.c_index,
                             v_index=self.v_index,
                             k_index=self.k_index,
                             other_index=self.other_index,
                             node_dropout_rate=self.node_dropout_rate,
                             mess_dropout_rate=self.mess_dropout_rate)

        self.feat_drop = nn.Dropout(self.mess_dropout_rate)
        self.indices = nn.Parameter(torch.arange(self.n_items, dtype=torch.long), requires_grad=False)
        self.gru = nn.GRU(self.emb_size, self.emb_size, batch_first=True)
        self.fc_self = nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.fc_neigh = nn.Linear(self.emb_size, self.emb_size, bias=False)
        
        input_dim = self.emb_size
        self.layers = nn.ModuleList()
        for i in range(self.context_hops):
            layer = GRUCov(
                input_dim,
                self.emb_size,
                self.mess_dropout_rate
            )
            self.layers.append(layer)
            input_dim += self.emb_size

        self.readout = Readout(
            input_dim,
            self.emb_size,
            self.emb_size,
            feat_drop=self.mess_dropout_rate
        )

        input_dim += self.emb_size
        self.fc_sr = nn.Linear(input_dim, self.emb_size, bias=False)
        self.activation = nn.PReLU(self.emb_size)
        self.gru = nn.GRU(self.emb_size, self.emb_size, batch_first=True)
        self.att_layer = AttentionLayer(att_hidden_size=[8 * self.emb_size, 256, 128, 64])
        self.fc_final = nn.Linear(self.emb_size * 7, self.emb_size)

    def forward(self, batch=None):
        """" Konwlege Concept Space """
        user_emb = self.all_emb[:self.n_users, :]
        item_emb = self.all_emb[self.n_users:, :]
        
        # entity_gcn_emb: [n_entity, emb_size]
        # user_gcn_emb: [n_users, emb_size]
        entity_gcn_emb, user_gcn_emb, cor = self.gcn(user_emb,
                                                     item_emb,
                                                     self.intent_emb,
                                                     self.edge_index,
                                                     self.edge_type,
                                                     self.interact_mat,
                                                     self.r_emb,
                                                     mess_dropout=self.mess_dropout,
                                                     node_dropout=self.node_dropout)
        
        """ Time-Space """
        user, ug, sub_ug, ssub_ug, seq, sub_seq, ssub_seq, lens, sub_lens, ssub_lens = batch
        u_e = user_gcn_emb[user]
        batch_size = u_e.shape[0]

        video_id = ug.ndata['video_id']
        video_feat = entity_gcn_emb[video_id]
        sub_video_id = sub_ug.ndata['video_id']
        sub_video_feat = entity_gcn_emb[sub_video_id]
        ssub_video_id = ssub_ug.ndata['video_id']
        ssub_video_feat = entity_gcn_emb[ssub_video_id]

        last_node = ug.filter_nodes(lambda nodes: nodes.data['last'] == 1)
        sub_last_node = sub_ug.filter_nodes(lambda nodes: nodes.data['last'] == 1)
        ssub_last_node = ssub_ug.filter_nodes(lambda nodes: nodes.data['last'] == 1)

        for layer in self.layers:
            out = layer(ug, video_feat, user, last_node)
            video_feat = torch.cat([out, video_feat], dim=1)
            sub_out = layer(sub_ug, sub_video_feat, user, sub_last_node)
            sub_video_feat = torch.cat([sub_out, sub_video_feat], dim=1)
            ssub_out = layer(ssub_ug, ssub_video_feat, user, ssub_last_node)
            ssub_video_feat = torch.cat([ssub_out, ssub_video_feat], dim=1)

        sr_g = self.readout(ug, video_feat, last_node)
        sr_l = video_feat[last_node]
        sr = torch.cat([sr_l, sr_g], dim=1)
        sr = self.fc_sr(self.feat_drop(sr))

        sub_sr_g = self.readout(sub_ug, sub_video_feat, sub_last_node)
        sub_sr_l = sub_video_feat[sub_last_node]
        sub_sr = torch.cat([sub_sr_l, sub_sr_g], dim=1)
        sub_sr = self.fc_sr(self.feat_drop(sub_sr))

        ssub_sr_g = self.readout(ssub_ug, ssub_video_feat, ssub_last_node)
        ssub_sr_l = ssub_video_feat[ssub_last_node]
        ssub_sr = torch.cat([ssub_sr_l, ssub_sr_g], dim=1)
        ssub_sr = self.fc_sr(self.feat_drop(ssub_sr))

        seq_emb = entity_gcn_emb[seq]
        gru_input = nn.utils.rnn.pack_padded_sequence(seq_emb, lens, batch_first=True, enforce_sorted=False)
        _, h = self.gru(gru_input)
        h = h.squeeze(0)
        
        sub_seq_emb = entity_gcn_emb[sub_seq]
        sub_gru_input = nn.utils.rnn.pack_padded_sequence(sub_seq_emb, sub_lens, batch_first=True, enforce_sorted=False)
        _, sub_h = self.gru(sub_gru_input)
        sub_h = sub_h.squeeze(0)

        ssub_seq_emb = entity_gcn_emb[ssub_seq]
        ssub_gru_input = nn.utils.rnn.pack_padded_sequence(ssub_seq_emb, ssub_lens, batch_first=True, enforce_sorted=False)
        _, ssub_h = self.gru(ssub_gru_input)
        ssub_h = ssub_h.squeeze(0)

        intents_kc = self.intent_emb.repeat(batch_size, 1).view(batch_size, self.n_intents, self.emb_size)
        ui_kc = self.att_layer(u_e, intents_kc, sr_l)

        intents_ts = ((torch.cat([sr, sub_sr, ssub_sr], dim=1) + torch.cat([h, sub_h, ssub_h], dim=1)) / 2).view(batch_size, 3, self.emb_size)
        ui_ts = self.att_layer(u_e, intents_ts, sr_l)
        input_tensor = torch.cat([u_e, ui_kc, ui_ts, sr_l], dim=1)
        fr = self.activation(self.fc_final(input_tensor))

        logits = fr @ entity_gcn_emb[:self.n_items, :].t()
        self_supervised_loss = self._create_self_supervised_loss(sr, sub_sr, ssub_sr, h, sub_h, ssub_h)
        return logits, cor, self_supervised_loss


    def predict(self, batch):
        """" Konwlege Concept Space """
        user_emb = self.all_emb[:self.n_users, :]
        item_emb = self.all_emb[self.n_users:, :]
        
        # entity_gcn_emb: [n_entity, emb_size]
        # user_gcn_emb: [n_users, emb_size]
        entity_gcn_emb, user_gcn_emb, cor = self.gcn(user_emb,
                                                     item_emb,
                                                     self.intent_emb,
                                                     self.edge_index,
                                                     self.edge_type,
                                                     self.interact_mat,
                                                     self.r_emb,
                                                     mess_dropout=self.mess_dropout,
                                                     node_dropout=self.node_dropout)
        
        """ Time-Space """
        user, ug, sub_ug, ssub_ug, seq, sub_seq, ssub_seq, lens, sub_lens, ssub_lens = batch
        u_e = user_gcn_emb[user]
        batch_size = u_e.shape[0]

        video_id = ug.ndata['video_id']
        video_feat = entity_gcn_emb[video_id]
        sub_video_id = sub_ug.ndata['video_id']
        sub_video_feat = entity_gcn_emb[sub_video_id]
        ssub_video_id = ssub_ug.ndata['video_id']
        ssub_video_feat = entity_gcn_emb[ssub_video_id]

        last_node = ug.filter_nodes(lambda nodes: nodes.data['last'] == 1)
        sub_last_node = sub_ug.filter_nodes(lambda nodes: nodes.data['last'] == 1)
        ssub_last_node = ssub_ug.filter_nodes(lambda nodes: nodes.data['last'] == 1)

        for layer in self.layers:
            out = layer(ug, video_feat, user, last_node)
            video_feat = torch.cat([out, video_feat], dim=1)
            sub_out = layer(sub_ug, sub_video_feat, user, sub_last_node)
            sub_video_feat = torch.cat([sub_out, sub_video_feat], dim=1)
            ssub_out = layer(ssub_ug, ssub_video_feat, user, ssub_last_node)
            ssub_video_feat = torch.cat([ssub_out, ssub_video_feat], dim=1)

        sr_g = self.readout(ug, video_feat, last_node)
        sr_l = video_feat[last_node]
        sr = torch.cat([sr_l, sr_g], dim=1)
        sr = self.fc_sr(self.feat_drop(sr))

        sub_sr_g = self.readout(sub_ug, sub_video_feat, sub_last_node)
        sub_sr_l = sub_video_feat[sub_last_node]
        sub_sr = torch.cat([sub_sr_l, sub_sr_g], dim=1)
        sub_sr = self.fc_sr(self.feat_drop(sub_sr))

        ssub_sr_g = self.readout(ssub_ug, ssub_video_feat, ssub_last_node)
        ssub_sr_l = ssub_video_feat[ssub_last_node]
        ssub_sr = torch.cat([ssub_sr_l, ssub_sr_g], dim=1)
        ssub_sr = self.fc_sr(self.feat_drop(ssub_sr))

        seq_emb = entity_gcn_emb[seq]
        seq_emb = seq_emb.unsqueeze(dim=0)
        gru_input = nn.utils.rnn.pack_padded_sequence(seq_emb, lens, batch_first=True, enforce_sorted=False)
        _, h = self.gru(gru_input)
        h = h.squeeze(0)
        
        sub_seq_emb = entity_gcn_emb[sub_seq]
        sub_seq_emb = sub_seq_emb.unsqueeze(dim=0)
        sub_gru_input = nn.utils.rnn.pack_padded_sequence(sub_seq_emb, sub_lens, batch_first=True, enforce_sorted=False)
        _, sub_h = self.gru(sub_gru_input)
        sub_h = sub_h.squeeze(0)

        ssub_seq_emb = entity_gcn_emb[ssub_seq]
        ssub_seq_emb = ssub_seq_emb.unsqueeze(dim=0)
        ssub_gru_input = nn.utils.rnn.pack_padded_sequence(ssub_seq_emb, ssub_lens, batch_first=True, enforce_sorted=False)
        _, ssub_h = self.gru(ssub_gru_input)
        ssub_h = ssub_h.squeeze(0)

        intents_kc = self.intent_emb.repeat(batch_size, 1).view(batch_size, self.n_intents, self.emb_size)
        ui_kc = self.att_layer(u_e, intents_kc, sr_l)

        intents_ts = ((torch.cat([sr, sub_sr, ssub_sr], dim=1) + torch.cat([h, sub_h, ssub_h], dim=1)) / 2).view(batch_size, 3, self.emb_size)
        ui_ts = self.att_layer(u_e, intents_ts, sr_l)
        input_tensor = torch.cat([u_e, ui_kc, ui_ts, sr_l], dim=1)
        fr = self.activation(self.fc_final(input_tensor))

        logits = fr @ entity_gcn_emb[:self.n_items, :].t()
        logits = logits.squeeze().detach().numpy()
        sorted_item_id = np.argsort(-logits)
        return sorted_item_id



    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # edge
        type = graph_tensor[:, -1]  # relation
        return index.t().long().to(self.device), type.long().to(self.device)

    def _create_self_supervised_loss(self, sr, sub_sr, ssub_sr, h, sub_h, ssub_h):
        pos_seq_score = torch.sum(torch.mul(sr, h), axis=1)
        pos_sub_seq_score = torch.sum(torch.mul(sub_sr, sub_h), axis=1)
        pos_ssub_seq_score = torch.sum(torch.mul(ssub_sr, ssub_h), axis=1)

        neg_seq_score_0 = torch.sum(torch.mul(sr, sub_sr), axis=1)
        neg_seq_score_1 = torch.sum(torch.mul(sr, ssub_sr), axis=1)
        neg_seq_score_2 = torch.sum(torch.mul(sub_sr, ssub_sr), axis=1)

        neg_seq_score_3 = torch.sum(torch.mul(h, sub_h), axis=1)
        neg_seq_score_4 = torch.sum(torch.mul(h, ssub_h), axis=1)
        neg_seq_score_5 = torch.sum(torch.mul(sub_h, ssub_h), axis=1)

        self_supervised_loss = -1 * torch.mean(nn.LogSigmoid()(pos_seq_score - neg_seq_score_0)) + \
                               -1 * torch.mean(nn.LogSigmoid()(pos_seq_score - neg_seq_score_1)) + \
                               -1 * torch.mean(nn.LogSigmoid()(pos_seq_score - neg_seq_score_3)) + \
                               -1 * torch.mean(nn.LogSigmoid()(pos_seq_score - neg_seq_score_4)) + \
                               -1 * torch.mean(nn.LogSigmoid()(pos_sub_seq_score - neg_seq_score_0)) + \
                               -1 * torch.mean(nn.LogSigmoid()(pos_sub_seq_score - neg_seq_score_2)) + \
                               -1 * torch.mean(nn.LogSigmoid()(pos_sub_seq_score - neg_seq_score_3)) + \
                               -1 * torch.mean(nn.LogSigmoid()(pos_sub_seq_score - neg_seq_score_5)) + \
                               -1 * torch.mean(nn.LogSigmoid()(pos_ssub_seq_score - neg_seq_score_1)) + \
                               -1 * torch.mean(nn.LogSigmoid()(pos_ssub_seq_score - neg_seq_score_2)) + \
                               -1 * torch.mean(nn.LogSigmoid()(pos_ssub_seq_score - neg_seq_score_4)) + \
                               -1 * torch.mean(nn.LogSigmoid()(pos_ssub_seq_score - neg_seq_score_5))
        return self_supervised_loss


        