import torch
import torch.nn as nn
from torch.nn import functional as F
import math


def make_fc(dim_in, hidden_dim):

    fc = nn.Linear(dim_in, hidden_dim)
    nn.init.kaiming_uniform_(fc.weight, a=1)
    nn.init.constant_(fc.bias, 0)
    return fc

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        # print("e.shape: ", e.shape) #[N, N]

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)   # 这句是关键
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

class AdjGenerator(nn.Module):
    def __init__(self, feat_dim):
        super(AdjGenerator, self).__init__()
        fcs, Wgs, Wqs, Wks, Wvs = [], [], [], [], []

        self.layernum = 3
        input_size = feat_dim
        representation_size = feat_dim

        self.embed_dim = 64
        self.groups = 16
        self.feat_dim = representation_size

        for i in range(self.layernum):
            r_size = input_size if i == 0 else representation_size

            if i != self.layernum:
                fcs.append(make_fc(r_size, representation_size))
            Wgs.append(nn.Conv2d(self.embed_dim, self.groups, kernel_size=1, stride=1, padding=0))
            Wqs.append(make_fc(self.feat_dim, self.feat_dim))
            Wks.append(make_fc(self.feat_dim, self.feat_dim))
            Wvs.append(nn.Conv2d(self.feat_dim * self.groups, self.feat_dim, kernel_size=1, stride=1, padding=0,
                              groups=self.groups))
            torch.nn.init.normal_(Wvs[i].weight, std=0.01)
            torch.nn.init.constant_(Wvs[i].bias, 0)

        self.fcs = nn.ModuleList(fcs)
        self.Wgs = nn.ModuleList(Wgs)
        self.Wqs = nn.ModuleList(Wqs)
        self.Wks = nn.ModuleList(Wks)
        self.Wvs = nn.ModuleList(Wvs)
        self.sigmoid = nn.Sigmoid()

    def attention_module_multi_head(self, roi_feat, ref_feat, position_embedding, feat_dim=2048, dim=(2048, 2048, 2048), group=16, index=0):
        """

        :param roi_feat: [num_rois, feat_dim]
        :param ref_feat: [num_nongt_rois, feat_dim]
        :param position_embedding: [1, emb_dim, num_rois, num_nongt_rois]
        :param feat_dim: should be same as dim[2]
        :param dim: a 3-tuple of (query, key, output)
        :param group:
        :return:
        """
        dim_group = (dim[0] / group, dim[1] / group, dim[2] / group)

        # position_embedding, [1, emb_dim, num_rois, num_nongt_rois]
        # -> position_feat_1, [1, group, num_rois, num_nongt_rois]   为了方便就把group设为1，不去改代码了
        position_feat_1 = F.relu(self.Wgs[index](position_embedding))
        # aff_weight, [num_rois, group, num_nongt_rois, 1]
        aff_weight = position_feat_1.permute(2, 1, 3, 0)
        # aff_weight, [num_rois, group, num_nongt_rois]
        aff_weight = aff_weight.squeeze(3)

        # multi head
        assert dim[0] == dim[1]
        
        # print(roi_feat.shape)
        q_data = self.Wqs[index](roi_feat) # [N, feature_dim=256]
        # print(q_data.shape)
        q_data_batch = q_data.reshape(-1, group, int(dim_group[0]))
        # q_data_batch, [group, num_rois, dim_group[0]]
        q_data_batch = q_data_batch.permute(1, 0, 2)
        # print(q_data_batch.shape)

        k_data = self.Wks[index](ref_feat)
        k_data_batch = k_data.reshape(-1, group, int(dim_group[1]))
        # k_data_batch, [group, num_nongt_rois, dim_group[1]]
        k_data_batch = k_data_batch.permute(1, 0, 2)

        # v_data, [num_nongt_rois, feat_dim]
        v_data = ref_feat

        # aff, [group, num_rois, num_nongt_rois]
        aff = torch.bmm(q_data_batch, k_data_batch.transpose(1, 2))
        aff_scale = (1.0 / math.sqrt(float(dim_group[1]))) * aff
        # aff_scale, [num_rois, group, num_nongt_rois]
        aff_scale = aff_scale.permute(1, 0, 2)

        # weighted_aff, [num_rois, group, num_nongt_rois]
        weighted_aff = (aff_weight + 1e-6).log() + aff_scale
        # aff_softmax = F.softmax(weighted_aff, dim=2)
        aff_sigmoid = self.sigmoid(weighted_aff).squeeze(1)   # 用sigmoid函数输出0-1的值
        # 取平均
        aff_sigmoid = torch.mean(aff_sigmoid, dim=1)
        adj = torch.where(aff_sigmoid > 0.5, 1., 0.)   # 大于0.5设为1，小于0.5设为0

        return adj

    def cal_position_embedding(self, rois1, rois2):
        # [num_rois, num_nongt_rois, 4, batchsize]
        position_matrix = self.extract_position_matrix(rois1, rois2)
        # [num_rois, num_nongt_rois, 64, batchsize]
        position_embedding = self.extract_position_embedding(position_matrix, feat_dim=64)
        # [batch_size, 64, num_rois, num_nongt_rois]
        position_embedding = position_embedding.permute(3, 2, 0, 1)
        # print(position_embedding.shape)
        # # [1, 64, num_rois, num_nongt_rois]
        # position_embedding = position_embedding.unsqueeze(0)

        return position_embedding

    def extract_position_matrix(self, bbox, ref_bbox):
        xmin, ymin, xmax, ymax = torch.chunk(ref_bbox, 4, dim=2)
        bbox_width_ref = xmax - xmin + 1
        bbox_height_ref = ymax - ymin + 1
        center_x_ref = 0.5 * (xmin + xmax)
        center_y_ref = 0.5 * (ymin + ymax)

        xmin, ymin, xmax, ymax = torch.chunk(bbox, 4, dim=2)
        bbox_width = xmax - xmin + 1
        bbox_height = ymax - ymin + 1
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)

        delta_x = center_x - center_x_ref.transpose(0, 1)
        delta_x = delta_x / bbox_width
        delta_x = (delta_x.abs() + 1e-3).log()

        delta_y = center_y - center_y_ref.transpose(0, 1)
        delta_y = delta_y / bbox_height
        delta_y = (delta_y.abs() + 1e-3).log()

        delta_width = bbox_width / bbox_width_ref.transpose(0, 1)
        delta_width = delta_width.log()

        delta_height = bbox_height / bbox_height_ref.transpose(0, 1)
        delta_height = delta_height.log()

        position_matrix = torch.stack([delta_x, delta_y, delta_width, delta_height], dim=2)

        return position_matrix

    def extract_position_embedding(self, position_mat, feat_dim, wave_length=1000.0):
        device = position_mat.device
        # position_mat, [num_rois, num_nongt_rois, 4, batch_size]
        feat_range = torch.arange(0, feat_dim / 8, device=device)
        dim_mat = torch.full((len(feat_range),), wave_length, device=device).pow(8.0 / feat_dim * feat_range)
        dim_mat = dim_mat.view(1, 1, 1, 1, -1).expand(*position_mat.shape, -1)  #[num_rois, num_nongt_rois, 4, batch_size, 8]

        dim_mat = dim_mat.permute(0,1,2,4,3)

        position_mat = position_mat.unsqueeze(4).expand(-1, -1, -1, dim_mat.shape[3], -1)
        position_mat = position_mat * 100.0

        div_mat = position_mat / dim_mat
        sin_mat, cos_mat = div_mat.sin(), div_mat.cos()

        # [num_rois, num_nongt_rois, 4, feat_dim / 4, batch_size]
        embedding = torch.cat([sin_mat, cos_mat], dim=3)
        # [num_rois, num_nongt_rois, feat_dim, batch_size]
        embedding = embedding.reshape(embedding.shape[0], embedding.shape[1], embedding.shape[2] * embedding.shape[3], -1)

        return embedding

    def forward(self, ref_feat, sup_feat, position_embedding, index=0):

        adj = self.attention_module_multi_head(ref_feat, sup_feat, position_embedding, feat_dim=self.feat_dim,
                                         dim=(self.feat_dim, self.feat_dim, self.feat_dim), group=1, index=index)

        return adj

class Relation(nn.Module):
    def __init__(self, feat_dim):
        super(Relation, self).__init__()
        fcs, Wgs, Wqs, Wks, Wvs = [], [], [], [], []

        self.layernum = 3
        input_size = feat_dim
        representation_size = feat_dim

        self.embed_dim = 64
        self.groups = 16
        self.feat_dim = representation_size

        for i in range(self.layernum):
            r_size = input_size if i == 0 else representation_size

            if i != self.layernum:
                fcs.append(make_fc(r_size, representation_size))
            Wgs.append(nn.Conv2d(self.embed_dim, self.groups, kernel_size=1, stride=1, padding=0))
            Wqs.append(make_fc(self.feat_dim, self.feat_dim))
            Wks.append(make_fc(self.feat_dim, self.feat_dim))
            Wvs.append(nn.Conv2d(self.feat_dim * self.groups, self.feat_dim, kernel_size=1, stride=1, padding=0,
                              groups=self.groups))
            torch.nn.init.normal_(Wvs[i].weight, std=0.01)
            torch.nn.init.constant_(Wvs[i].bias, 0)

        self.fcs = nn.ModuleList(fcs)
        self.Wgs = nn.ModuleList(Wgs)
        self.Wqs = nn.ModuleList(Wqs)
        self.Wks = nn.ModuleList(Wks)
        self.Wvs = nn.ModuleList(Wvs)

    def attention_module_multi_head(self, roi_feat, ref_feat, position_embedding, feat_dim=2048, dim=(2048, 2048, 2048), group=16, index=0):
        """

        :param roi_feat: [num_rois, feat_dim]
        :param ref_feat: [num_nongt_rois, feat_dim]
        :param position_embedding: [1, emb_dim, num_rois, num_nongt_rois]
        :param feat_dim: should be same as dim[2]
        :param dim: a 3-tuple of (query, key, output)
        :param group:
        :return:
        """
        dim_group = (dim[0] / group, dim[1] / group, dim[2] / group)

        # position_embedding, [1, emb_dim, num_rois, num_nongt_rois]
        # -> position_feat_1, [1, group, num_rois, num_nongt_rois]
        position_feat_1 = F.relu(self.Wgs[index](position_embedding))
        # aff_weight, [num_rois, group, num_nongt_rois, 1]
        aff_weight = position_feat_1.permute(2, 1, 3, 0)
        # aff_weight, [num_rois, group, num_nongt_rois]
        aff_weight = aff_weight.squeeze(3)

        # multi head
        assert dim[0] == dim[1]

        q_data = self.Wqs[index](roi_feat)
        q_data_batch = q_data.reshape(-1, group, int(dim_group[0]))
        # q_data_batch, [group, num_rois, dim_group[0]]
        q_data_batch = q_data_batch.permute(1, 0, 2)

        k_data = self.Wks[index](ref_feat)
        k_data_batch = k_data.reshape(-1, group, int(dim_group[1]))
        # k_data_batch, [group, num_nongt_rois, dim_group[1]]
        k_data_batch = k_data_batch.permute(1, 0, 2)

        # v_data, [num_nongt_rois, feat_dim]
        v_data = ref_feat

        # aff, [group, num_rois, num_nongt_rois]
        aff = torch.bmm(q_data_batch, k_data_batch.transpose(1, 2))
        aff_scale = (1.0 / math.sqrt(float(dim_group[1]))) * aff
        # aff_scale, [num_rois, group, num_nongt_rois]
        aff_scale = aff_scale.permute(1, 0, 2)

        # weighted_aff, [num_rois, group, num_nongt_rois]
        weighted_aff = (aff_weight + 1e-6).log() + aff_scale
        aff_softmax = F.softmax(weighted_aff, dim=2)

        aff_softmax_reshape = aff_softmax.reshape(aff_softmax.shape[0] * aff_softmax.shape[1], aff_softmax.shape[2])

        # output_t, [num_rois * group, feat_dim]
        output_t = torch.matmul(aff_softmax_reshape, v_data)
        # output_t, [num_rois, group * feat_dim, 1, 1]
        output_t = output_t.reshape(-1, group * feat_dim, 1, 1)
        # linear_out, [num_rois, dim[2], 1, 1]
        linear_out = self.Wvs[index](output_t)

        output = linear_out.squeeze(3).squeeze(2)

        return output

    def cal_position_embedding(self, rois1, rois2):
        # [num_rois, num_nongt_rois, 4]
        position_matrix = self.extract_position_matrix(rois1, rois2)
        # [num_rois, num_nongt_rois, 64]
        position_embedding = self.extract_position_embedding(position_matrix, feat_dim=64)
        # [64, num_rois, num_nongt_rois]
        position_embedding = position_embedding.permute(2, 0, 1)
        # [1, 64, num_rois, num_nongt_rois]
        position_embedding = position_embedding.unsqueeze(0)

        return position_embedding

    def extract_position_matrix(bbox, ref_bbox):
        xmin, ymin, xmax, ymax = torch.chunk(ref_bbox, 4, dim=1)
        bbox_width_ref = xmax - xmin + 1
        bbox_height_ref = ymax - ymin + 1
        center_x_ref = 0.5 * (xmin + xmax)
        center_y_ref = 0.5 * (ymin + ymax)

        xmin, ymin, xmax, ymax = torch.chunk(bbox, 4, dim=1)
        bbox_width = xmax - xmin + 1
        bbox_height = ymax - ymin + 1
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)

        delta_x = center_x - center_x_ref.transpose(0, 1)
        delta_x = delta_x / bbox_width
        delta_x = (delta_x.abs() + 1e-3).log()

        delta_y = center_y - center_y_ref.transpose(0, 1)
        delta_y = delta_y / bbox_height
        delta_y = (delta_y.abs() + 1e-3).log()

        delta_width = bbox_width / bbox_width_ref.transpose(0, 1)
        delta_width = delta_width.log()

        delta_height = bbox_height / bbox_height_ref.transpose(0, 1)
        delta_height = delta_height.log()

        position_matrix = torch.stack([delta_x, delta_y, delta_width, delta_height], dim=2)

        return position_matrix

    def extract_position_embedding(position_mat, feat_dim, wave_length=1000.0):
        device = position_mat.device
        # position_mat, [num_rois, num_nongt_rois, 4]
        feat_range = torch.arange(0, feat_dim / 8, device=device)
        dim_mat = torch.full((len(feat_range),), wave_length, device=device).pow(8.0 / feat_dim * feat_range)
        dim_mat = dim_mat.view(1, 1, 1, -1).expand(*position_mat.shape, -1)

        position_mat = position_mat.unsqueeze(3).expand(-1, -1, -1, dim_mat.shape[3])
        position_mat = position_mat * 100.0

        div_mat = position_mat / dim_mat
        sin_mat, cos_mat = div_mat.sin(), div_mat.cos()

        # [num_rois, num_nongt_rois, 4, feat_dim / 4]
        embedding = torch.cat([sin_mat, cos_mat], dim=3)
        # [num_rois, num_nongt_rois, feat_dim]
        embedding = embedding.reshape(embedding.shape[0], embedding.shape[1], embedding.shape[2] * embedding.shape[3])

        return embedding

    def forward(self, ref_feat, sup_feat, position_embedding, index=0):
        # feat: [B, N, F]
        # ref_feat = F.relu(self.fcs[index](ref_feat))
        # sup_feat = F.relu(self.fcs[0](sup_feat))
        #
        # attention = self.attention_module_multi_head(ref_feat, sup_feat, index=index)
        # ref_feat = ref_feat + attention

        sup_feat = F.relu(self.fcs[0](sup_feat))
        for i in range(1):
            ref_feat = F.relu(self.fcs[i](ref_feat))
            attention = self.attention_module_multi_head(ref_feat, sup_feat, position_embedding, feat_dim=self.feat_dim, dim=(self.feat_dim, self.feat_dim, self.feat_dim), index=i)
            ref_feat = ref_feat + attention

        return ref_feat


if __name__ == '__main__':
    input = torch.rand([32,8,2048], dtype=torch.float)
