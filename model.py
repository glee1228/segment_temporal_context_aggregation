import copy
import math
import numpy as np
import horovod.torch as hvd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models

#
# class NetVLAD(nn.Module):
#     """NetVLAD layer implementation"""
#
#     def __init__(self, feature_size, num_clusters=256, output_dim=1024, normalize_input=True, alpha=1.0, drop_rate=0.5, gating_reduction=8):
#         super(NetVLAD, self).__init__()
#         self.feature_size = feature_size
#         self.num_clusters = num_clusters
#         self.normalize_input = normalize_input
#         self.alpha = alpha
#
#         self.bn1 = nn.BatchNorm1d(feature_size)
#         self.conv = nn.Conv1d(feature_size, num_clusters,
#                               kernel_size=1, bias=True)
#         self.centroids = nn.Parameter(torch.rand(num_clusters, feature_size))
#
#         self.bn2 = nn.BatchNorm1d(feature_size * num_clusters)
#         self.drop = nn.Dropout(drop_rate)
#
#         self.fc1 = nn.Linear(feature_size * num_clusters, output_dim)
#         self.bn3 = nn.BatchNorm1d(output_dim)
#         self.fc2 = nn.Linear(output_dim, output_dim // gating_reduction)
#         self.bn4 = nn.BatchNorm1d(output_dim // gating_reduction)
#         self.fc3 = nn.Linear(output_dim // gating_reduction, output_dim)
#         self._init_params()
#
#     def _init_params(self):
#         self.conv.weight = nn.Parameter(
#             (2.0 * self.alpha * self.centroids).unsqueeze(-1)
#         )
#         self.conv.bias = nn.Parameter(
#             - self.alpha * self.centroids.norm(dim=1)
#         )
#
#     def forward(self, x, num_frames):
#         N, C, T = x.shape[:3]  # (N, C, T)
#         # mask padded frame feature
#         if len(num_frames.shape) == 1:
#             num_frames = num_frames.unsqueeze(1)
#         frame_mask = (
#             0 < num_frames - torch.arange(0, T).cuda()
#         ).float()  # (N, T)
#
#         assert C == self.feature_size, 'Input should have feature_size {} but got {}.'.format(self.feature_size, C)
#
#         x = self.bn1(x)
#         if self.normalize_input:
#             x = F.normalize(x, p=2, dim=1)  # across descriptor dim
#
#         # soft-assignment
#         soft_assign = self.conv(x)  # (N, num_clusters, T)
#         soft_assign = F.softmax(soft_assign, dim=1)  # (N, num_clusters, T)
#         soft_assign = soft_assign * frame_mask.unsqueeze(1)
#
#         soft_assign_sum = torch.sum(
#             soft_assign, dim=-1, keepdim=True)  # (N, num_clusters, 1)
#         # (N, num_clusters, feature_size)
#         centervlad = self.centroids * soft_assign_sum
#
#         x_flatten = x.view(N, C, -1)  # (N, feature_size, T)
#         # (N, num_clusters, feature_size)
#         vlad = torch.bmm(soft_assign, x_flatten.transpose(1, 2))
#         vlad -= centervlad  # (N, num_clusters, feature_size)
#
#         # intra-normalization (N, num_clusters, feature_size)
#         vlad = F.normalize(vlad, p=2, dim=2)
#         # flatten (N, num_clusters * feature_size)
#         vlad = vlad.view(x.size(0), -1)
#         vlad = self.bn2(vlad)
#
#         vlad = self.drop(vlad)
#
#         activation = self.bn3(self.fc1(vlad))  # (N, output_dim)
#
#         # (N, output_dim // gating_reduction)
#         gates = F.relu(self.bn4(self.fc2(activation)))
#         gates = torch.sigmoid(self.fc3(gates))  # (N, output_dim)
#
#         activation = activation * gates  # (N, output_dim)
#         # L2 normalize (N, num_clusters * feature_size)
#         vlad = F.normalize(activation, p=2, dim=1)
#
#         # L2 normalize (N, output_dim) IMPORTANT!!!
#         embedding = F.normalize(vlad, p=2, dim=1)
#
#         return embedding  # (N, output_dim)
#
#     def encode(self, x, num_frames):
#         N, C, T = x.shape[:3]  # (N, C, T)
#         # mask padded frame feature
#         if len(num_frames.shape) == 1:
#             num_frames = num_frames.unsqueeze(1)
#         frame_mask = (
#             0 < num_frames - torch.arange(0, T).cuda()
#         ).float()  # (N, T)
#
#         assert C == self.feature_size, 'Input should have feature_size {} but got {}.'.format(self.feature_size, C)
#         x = self.bn1(x)
#         if self.normalize_input:
#             x = F.normalize(x, p=2, dim=1)  # across descriptor dim
#
#         # soft-assignment
#         soft_assign = self.conv(x)  # (N, num_clusters, T)
#         soft_assign = F.softmax(soft_assign, dim=1)  # (N, num_clusters, T)
#         soft_assign = soft_assign * frame_mask.unsqueeze(1)
#
#         soft_assign_sum = torch.sum(
#             soft_assign, dim=-1, keepdim=True)  # (N, num_clusters, 1)
#         # (N, num_clusters, feature_size)
#         centervlad = self.centroids * soft_assign_sum
#
#         x_flatten = x.view(N, C, -1)  # (N, feature_size, T)
#         # (N, num_clusters, feature_size)
#         vlad = torch.bmm(soft_assign, x_flatten.transpose(1, 2))
#         vlad -= centervlad  # (N, num_clusters, feature_size)
#
#         # intra-normalization (N, num_clusters, feature_size)
#         vlad = F.normalize(vlad, p=2, dim=2)
#         return vlad

class NetVLAD(nn.Module):

    def __init__(self, dims, num_clusters, outdims=None):
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dims = dims

        self.centroids = nn.Parameter(torch.randn(num_clusters, dims) / math.sqrt(self.dims))
        self.conv = nn.Conv2d(dims, num_clusters, kernel_size=1, bias=False)

        if outdims is not None:
            self.outdims = outdims
            self.reduction_layer = nn.Linear(self.num_clusters * self.dims, self.outdims, bias=False)
        else:
            self.outdims = self.num_clusters * self.dims
        self.norm = nn.LayerNorm(self.outdims)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.weight = nn.Parameter(self.centroids.detach().clone().unsqueeze(-1).unsqueeze(-1))
        if hasattr(self, 'reduction_layer'):
            nn.init.normal_(self.reduction_layer.weight, std=1 / math.sqrt(self.num_clusters * self.dims))

    def forward(self, x, mask=None, sample=False):
        N, C, T, R = x.shape

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1).view(N, self.num_clusters, T, R)

        x_flatten = x.view(N, C, -1)

        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for cluster in range(self.num_clusters):  # slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - self.centroids[cluster:cluster + 1, :]. \
                expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual = residual.view(N, C, T, R)
            residual *= soft_assign[:, cluster:cluster + 1, :]
            if mask is not None:
                residual = residual.masked_fill((1 - mask.unsqueeze(1).unsqueeze(-1)).bool(), 0.0)
            vlad[:, cluster:cluster + 1, :] = residual.sum([-2, -1]).unsqueeze(1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        if hasattr(self, 'reduction_layer'):
            vlad = self.reduction_layer(vlad)
        return self.norm(vlad)

class NeXtVLAD(nn.Module):
    """NeXtVLAD layer implementation"""

    def __init__(self, feature_size, num_clusters=64, output_dim=1024, normalize_input=True, expansion=2, groups=8, drop_rate=0.5, gating_reduction=8):
        super(NeXtVLAD, self).__init__()
        self.feature_size = feature_size
        self.num_clusters = num_clusters
        self.normalize_input = normalize_input
        self.expansion = expansion
        self.groups = groups

        self.conv1 = nn.Conv1d(
            feature_size, feature_size * expansion, kernel_size=1, bias=True)
        self.conv2 = nn.Conv1d(feature_size * expansion,
                               groups, kernel_size=1, bias=True)
        self.conv3 = nn.Conv1d(feature_size * expansion,
                               num_clusters * groups, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(num_clusters * groups)
        self.centroids = nn.Parameter(torch.rand(
            num_clusters, feature_size * expansion // groups))

        self.bn2 = nn.BatchNorm1d(
            feature_size * expansion // groups * num_clusters)
        self.drop = nn.Dropout(drop_rate)
        self.fc1 = nn.Linear(feature_size * expansion //
                             groups * num_clusters, output_dim)
        self.bn3 = nn.BatchNorm1d(output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim // gating_reduction)
        self.bn4 = nn.BatchNorm1d(output_dim // gating_reduction)
        self.fc3 = nn.Linear(output_dim // gating_reduction, output_dim)

    def forward(self, x, num_frames):
        N, C, T = x.shape[:3]  # (N, C, T)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)
        assert C == self.feature_size, 'Input should have feature_size {} but got {}.'.format(self.feature_size, C)

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        x = self.conv1(x)  # (N, feature_size * expansion, T)
        # attention factor of per group
        attention = torch.sigmoid(self.conv2(x))  # (N, groups, T)
        attention = attention * frame_mask.unsqueeze(1)
        attention = attention.view(N, 1, -1)  # (N, 1, groups * T)
        # calculate activation factor of per group per cluster
        feature_size = self.feature_size * self.expansion // self.groups

        activation = self.conv3(x)  # (N, num_clusters * groups, T)
        activation = self.bn1(activation)
        # reshape of activation
        # (N, num_clusters, groups * T)
        activation = activation.view(N, self.num_clusters, -1)
        # softmax on per cluster
        # (N, num_clusters, groups * T)
        activation = F.softmax(activation, dim=1)
        activation = activation * attention  # (N, num_clusters, groups * T)
        activation_sum = torch.sum(
            activation, dim=-1, keepdim=True)  # (N, num_clusters, 1)
        # (N, num_clusters, feature_size)
        centervlad = self.centroids * activation_sum

        # (N, feature_size, groups * T)
        x_rehaped = x.view(N, feature_size, -1)
        vlad = torch.bmm(activation, x_rehaped.transpose(1, 2)
                         )  # (N, num_clusters, feature_size)
        vlad -= centervlad  # (N, num_clusters, feature_size)

        # intra-normalization (N, num_clusters, feature_size)
        vlad = F.normalize(vlad, p=2, dim=2)
        # flatten (N, num_clusters * feature_size)
        vlad = vlad.view(N, -1)
        vlad = self.bn2(vlad)

        vlad = self.drop(vlad)

        activation = self.bn3(self.fc1(vlad))  # (N, output_dim)

        # (N, output_dim // gating_reduction)
        gates = F.relu(self.bn4(self.fc2(activation)))
        gates = torch.sigmoid(self.fc3(gates))  # (N, output_dim)

        activation = activation * gates  # (N, output_dim)

        # L2 normalize (N, output_dim) IMPORTANT!!!
        embedding = F.normalize(activation, p=2, dim=1)

        return embedding  # (N, output_dim)

    def encode(self, x, num_frames):
        N, C, T = x.shape[:3]  # (N, C, T)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)
        assert C == self.feature_size, 'Input should have feature_size {} but got {}.'.format(self.feature_size, C)

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        x = self.conv1(x)  # (N, feature_size * expansion, T)
        # attention factor of per group
        attention = torch.sigmoid(self.conv2(x))  # (N, groups, T)
        attention = attention * frame_mask.unsqueeze(1)
        attention = attention.view(N, 1, -1)  # (N, 1, groups * T)
        # calculate activation factor of per group per cluster
        feature_size = self.feature_size * self.expansion // self.groups

        activation = self.conv3(x)  # (N, num_clusters * groups, T)
        activation = self.bn1(activation)
        # reshape of activation
        # (N, num_clusters, groups * T)
        activation = activation.view(N, self.num_clusters, -1)
        # softmax on per cluster
        # (N, num_clusters, groups * T)
        activation = F.softmax(activation, dim=1)
        activation = activation * attention  # (N, num_clusters, groups * T)
        activation_sum = torch.sum(
            activation, dim=-1, keepdim=True)  # (N, num_clusters, 1)
        # (N, num_clusters, feature_size)
        centervlad = self.centroids * activation_sum

        # (N, feature_size, groups * T)
        x_rehaped = x.view(N, feature_size, -1)
        vlad = torch.bmm(activation, x_rehaped.transpose(1, 2)
                         )  # (N, num_clusters, feature_size)
        vlad -= centervlad  # (N, num_clusters, feature_size)

        # intra-normalization (N, num_clusters, feature_size)
        vlad = F.normalize(vlad, p=2, dim=2)
        return vlad


class LSTMModule(nn.Module):
    def __init__(self, feature_size=1024, output_dim=1024, nhid=1024, nlayers=2, dropout=0.2):
        super(LSTMModule, self).__init__()

        self.feature_size = feature_size
        self.nhid = nhid
        self.nlayers = nlayers
        self.output_dim = output_dim
        self.dropout = dropout

        self.LSTM = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=self.nhid,
            num_layers=self.nlayers,
            dropout=self.dropout,
            # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            batch_first=True,
        )

        self.bn1 = nn.BatchNorm1d(self.feature_size)

    def forward(self, x, num_frames):
        x = self.bn1(x)
        x = x.transpose(-1, -2)
        N, T, C = x.shape[:3]  # (N, T, C)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)

        assert C == self.feature_size, 'Input should have feature_size {} but got {}.'.format(self.feature_size, C)

        self.LSTM.flatten_parameters()
        output, (h_n, h_c) = self.LSTM(x, None)

        output = output * frame_mask.unsqueeze(-1)
        frame_count = torch.sum(frame_mask, dim=-1, keepdim=True)  # (batch, 1)
        output = torch.sum(output, dim=-2) / frame_count

        # L2 normalize (N, output_dim) IMPORTANT!!!
        embedding = F.normalize(output, p=2, dim=1)
        return embedding

    def encode(self, x, num_frames):
        x = self.bn1(x)  # (N, C, T)
        x = x.transpose(-1, -2)
        N, T, C = x.shape[:3]  # (N, T, C)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)
        assert C == self.feature_size, 'Input should have feature_size {} but got {}.'.format(self.feature_size, C)

        self.LSTM.flatten_parameters()
        output, (h_n, h_c) = self.LSTM(x, None)
        output = output * frame_mask.unsqueeze(-1)
        frame_count = torch.sum(frame_mask, dim=-1)  # (N)
        output = torch.narrow(output, 1, 0, int(frame_count.item()))
        # L2 normalize IMPORTANT!!!
        output = F.normalize(output, p=2, dim=2)  # (N, T, C)
        return output


class GRUModule(nn.Module):
    def __init__(self, feature_size=1024, output_dim=1024, nhid=1024, nlayers=2, dropout=0.2):
        super(GRUModule, self).__init__()

        self.feature_size = feature_size
        self.nhid = nhid
        self.nlayers = nlayers
        self.output_dim = output_dim
        self.dropout = dropout

        self.GRU = nn.GRU(
            input_size=self.feature_size,
            hidden_size=self.nhid,
            num_layers=self.nlayers,
            dropout=self.dropout,
            # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            batch_first=True,
        )

        self.bn1 = nn.BatchNorm1d(self.feature_size)

    def forward(self, x, num_frames):
        x = self.bn1(x)
        x = x.transpose(-1, -2)
        N, T, C = x.shape[:3]  # (N, T, C)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)

        assert C == self.feature_size, 'Input should have feature_size {} but got {}.'.format(self.feature_size, C)
        self.GRU.flatten_parameters()
        output, h_n = self.GRU(x, None)

        output = output * frame_mask.unsqueeze(-1)
        frame_count = torch.sum(frame_mask, dim=-1, keepdim=True)  # (batch, 1)
        output = torch.sum(output, dim=-2) / frame_count

        # L2 normalize (N*num_directions, output_dim) IMPORTANT!!!
        embedding = F.normalize(output, p=2, dim=1)
        return embedding

    def encode(self, x, num_frames):
        x = self.bn1(x)  # (N, C, T)
        x = x.transpose(-1, -2)
        N, T, C = x.shape[:3]  # (N, T, C)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)
        assert C == self.feature_size, 'Input should have feature_size {} but got {}.'.format(self.feature_size, C)
        self.GRU.flatten_parameters()
        output, (h_n, h_c) = self.GRU(x, None)
        output = output * frame_mask.unsqueeze(-1)
        frame_count = torch.sum(frame_mask, dim=-1)  # (N)
        output = torch.narrow(output, 1, 0, int(frame_count.item()))
        # L2 normalize IMPORTANT!!!
        output = F.normalize(output, p=2, dim=2)  # (N, T, C)
        return output


class FSTA(nn.Module):
    def __init__(self, frame_feature_size=1024,segment_feature_size=2048, max_seq_len=128, nhead=8, nlayers=1, dropout=0.1):
        super(FSTA, self).__init__()

        self.frame_feature_size = frame_feature_size
        self.segment_feature_size = segment_feature_size
        self.max_seq_len = max_seq_len
        self.nhead = nhead
        self.nhid = nlayers
        self.dropout = dropout

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=frame_feature_size,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=dropout
        )
        encoder_layers2 = nn.TransformerEncoderLayer(
            d_model=segment_feature_size,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=dropout
        )
        self.frame_transformer_encoder = nn.TransformerEncoder(
            encoder_layers, nlayers)

        self.segment_transformer_encoder = nn.TransformerEncoder(
            encoder_layers2, nlayers)
        self.mlp = None

    def forward(self, x, num_frames):
        x = x.permute(2, 0, 1)  # (T, N, C) -> e.g., (300, 64, 1024)

        x, x2 = torch.split(x, int(x.shape[2]/2),dim=2)

        T, N, C = x.shape[:3]  # (T, N, C)
        T2, N2, C2 = x2.shape[:3]  # (T, N, C)

        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)

        assert C == self.frame_feature_size, 'Input should have feature_size {} but got {}.'.format(self.frame_feature_size, C)

        output = self.frame_transformer_encoder(
            x, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)

        output = torch.cat((output,x2),2)
        output = self.segment_transformer_encoder(
            output, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)
        output = output.permute(1, 0, 2)  # (N, T, C)
        output = output * frame_mask.unsqueeze(-1)
        frame_count = torch.sum(frame_mask, dim=-1, keepdim=True)  # (N, 1)
        output = torch.sum(output, dim=-2) / frame_count  # (N, C)

        # L2 normalize (N, output_dim) IMPORTANT!!!
        embedding = F.normalize(output, p=2, dim=1)
        if self.mlp is not None:
            embedding = self.mlp(embedding)
        return embedding

    def encode(self, x, x2, num_frames):
        x = x.permute(2, 0, 1)  # (T, N, C) -> e.g., (300, 64, 1024)
        x, x2 = torch.split(x, int(x.shape[2]/2),dim=2)
        T, N, C = x.shape[:3]  # (T, N, C)
        T2, N2, C2 = x2.shape[:3]  # (T, N, C)

        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)

        assert C == self.frame_feature_size, 'Input should have frame_feature_size {} but got {}.'.format(self.frame_feature_size, C)
        assert C2 == self.segment_feature_size, 'Input should have segment_feature_size {} but got {}.'.format(self.segment_feature_size, C2)

        output = self.frame_transformer_encoder(
            x, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)
        output = torch.cat((output, x2), 2)
        output = self.segment_transformer_encoder(
            output, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)
        output = output.permute(1, 0, 2)  # (N, T, C)
        output = output * frame_mask.unsqueeze(-1)
        frame_count = torch.sum(frame_mask, dim=-1)  # (N)
        output = torch.narrow(output, 1, 0, int(frame_count.item()))

        # L2 normalize IMPORTANT!!!
        output = F.normalize(output, p=2, dim=2)  # (N, T, C)
        if self.mlp is not None:
            output = self.mlp(output)
        return output


class CTCA(nn.Module):
    def __init__(self, feature_size=1024, feedforward=4096 , max_seq_len=128, nhead=8, nlayers=1, dropout=0.1):
        super(CTCA, self).__init__()

        self.feature_size = feature_size
        self.max_seq_len = max_seq_len
        self.nhead = nhead
        self.nhid = nlayers
        self.dropout = dropout

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=feature_size,
            nhead=nhead,
            dim_feedforward=feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, nlayers)
        self.mlp = None

    def forward(self, x, num_frames):
        x = x.permute(2, 0, 1)  # (T, N, C) -> e.g., (300, 64, 1024)

        T, N, C = x.shape[:3]  # (T, N, C)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)
        # print(frame_mask.shape)
        assert C == self.feature_size, 'Input should have feature_size {} but got {}.'.format(self.feature_size, C)
        # print(x.shape)
        output = self.transformer_encoder(
            x, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)
        # print(output.shape)
        output = output.permute(1, 0, 2)  # (N, T, C)
        # print(output.shape)
        output = output * frame_mask.unsqueeze(-1)
        # print(output.shape)
        frame_count = torch.sum(frame_mask, dim=-1, keepdim=True)  # (N, 1)
        # print(frame_count.shape)
        output = torch.sum(output, dim=-2) / frame_count  # (N, C)
        # print(output.shape)
        # L2 normalize (N, output_dim) IMPORTANT!!!
        embedding = F.normalize(output, p=2, dim=1)
        # breakpoint()
        if self.mlp is not None:
            embedding = self.mlp(embedding)
        return embedding

    def encode(self, x, num_frames):
        x = x.permute(2, 0, 1)  # (T, N, C)

        T, N, C = x.shape[:3]  # (T, N, C)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)
        assert C == self.feature_size, 'Input should have feature_size {} but got {}.'.format(self.feature_size, C)

        output = self.transformer_encoder(
            x, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)
        output = output.permute(1, 0, 2)  # (N, T, C)
        output = output * frame_mask.unsqueeze(-1)
        frame_count = torch.sum(frame_mask, dim=-1)  # (N)
        output = torch.narrow(output, 1, 0, int(frame_count.item()))

        # L2 normalize IMPORTANT!!!
        output = F.normalize(output, p=2, dim=2)  # (N, T, C)
        if self.mlp is not None:
            output = self.mlp(output)
        return output


class CTCA_PLUS(nn.Module):
    def __init__(self, feature_size=2048, feedforward=4096, max_seq_len=128, nhead=8, nlayers=1, dropout=0.1):
        super(CTCA_PLUS, self).__init__()

        self.feature_size = feature_size
        self.max_seq_len = max_seq_len
        self.nhead = nhead
        self.nhid = nlayers
        self.dropout = dropout

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=int(feature_size/2),
            nhead=nhead,
            dim_feedforward=feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, nlayers)
        self.mlp = None

    def forward(self, x, num_frames):
        x = x.permute(2, 0, 1)  # (T, N, C) -> e.g., (300, 64, 1024)

        T, N, C = x.shape[:3]  # (T, N, C)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)
        # print(frame_mask.shape)
        assert C == self.feature_size, 'Input should have feature_size {} but got {}.'.format(self.feature_size, C)
        # print(x.shape)
        x_f, x_t = x.split(self.frame_feature_size, dim=2)
        x = x_f + x_t
        output = self.transformer_encoder(
            x, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)
        # print(output.shape)
        output = output.permute(1, 0, 2)  # (N, T, C)
        # print(output.shape)
        output = output * frame_mask.unsqueeze(-1)
        # print(output.shape)
        frame_count = torch.sum(frame_mask, dim=-1, keepdim=True)  # (N, 1)
        # print(frame_count.shape)
        output = torch.sum(output, dim=-2) / frame_count  # (N, C)
        # print(output.shape)
        # L2 normalize (N, output_dim) IMPORTANT!!!
        embedding = F.normalize(output, p=2, dim=1)
        # breakpoint()
        if self.mlp is not None:
            embedding = self.mlp(embedding)
        return embedding

class CTCA_MAX(nn.Module):
    def __init__(self, feature_size=2048, feedforward=4096, max_seq_len=128, nhead=8, nlayers=1, dropout=0.1):
        super(CTCA_MAX, self).__init__()

        self.feature_size = feature_size
        self.max_seq_len = max_seq_len
        self.nhead = nhead
        self.nhid = nlayers
        self.dropout = dropout

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=int(feature_size/2),
            nhead=nhead,
            dim_feedforward=feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, nlayers)
        self.mlp = None

    def forward(self, x, num_frames):
        x = x.permute(2, 0, 1)  # (T, N, C) -> e.g., (300, 64, 1024)

        T, N, C = x.shape[:3]  # (T, N, C)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)
        # print(frame_mask.shape)
        assert C == self.feature_size, 'Input should have feature_size {} but got {}.'.format(self.feature_size, C)
        # print(x.shape)
        x_f, x_t = x.split(int(self.feature_size/2), dim=2)
        x = torch.maximum(x_f,x_t)

        output = self.transformer_encoder(
            x, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)
        # print(output.shape)
        output = output.permute(1, 0, 2)  # (N, T, C)
        # print(output.shape)
        output = output * frame_mask.unsqueeze(-1)
        # print(output.shape)
        frame_count = torch.sum(frame_mask, dim=-1, keepdim=True)  # (N, 1)
        # print(frame_count.shape)
        output = torch.sum(output, dim=-2) / frame_count  # (N, C)
        # print(output.shape)
        # L2 normalize (N, output_dim) IMPORTANT!!!
        embedding = F.normalize(output, p=2, dim=1)
        # breakpoint()
        if self.mlp is not None:
            embedding = self.mlp(embedding)
        return embedding
class CTCA_FC(nn.Module):
    def __init__(self, feature_size=1024, feedforward=4096 , max_seq_len=128, nhead=8, nlayers=1, dropout=0.1):
        super(CTCA_FC, self).__init__()

        self.feature_size = feature_size
        self.max_seq_len = max_seq_len
        self.nhead = nhead
        self.nhid = nlayers
        self.dropout = dropout
        self.linear = nn.Linear(feature_size, 1024, bias=False)
        self.norm = nn.LayerNorm(1024)
        self.activation = nn.ReLU()
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=feature_size,
            nhead=nhead,
            dim_feedforward=feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, nlayers, nn.Sequential(self.linear, self.activation, self.norm))
        self.mlp = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear.weight, std=1 / math.sqrt(1024))

    def forward(self, x, num_frames):
        x = x.permute(2, 0, 1)  # (T, N, C) -> e.g., (300, 64, 1024)

        T, N, C = x.shape[:3]  # (T, N, C)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)
        # print(frame_mask.shape)
        assert C == self.feature_size, 'Input should have feature_size {} but got {}.'.format(self.feature_size, C)
        # print(x.shape)
        output = self.transformer_encoder(
            x, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)
        # print(output.shape)
        output = output.permute(1, 0, 2)  # (N, T, C)
        # print(output.shape)
        output = output * frame_mask.unsqueeze(-1)
        # print(output.shape)
        frame_count = torch.sum(frame_mask, dim=-1, keepdim=True)  # (N, 1)
        # print(frame_count.shape)
        output = torch.sum(output, dim=-2) / frame_count  # (N, C)
        # print(output.shape)
        # L2 normalize (N, output_dim) IMPORTANT!!!
        embedding = F.normalize(output, p=2, dim=1)
        # breakpoint()
        if self.mlp is not None:
            embedding = self.mlp(embedding)
        return embedding


class CTCA_LATE(nn.Module):
    def __init__(self, frame_feature_size=1024, temporal_feature_size=1024, feedforward=2048, max_seq_len=128, nhead=8, nlayers=1, dropout=0.1):
        super(CTCA_LATE, self).__init__()

        self.frame_feature_size = frame_feature_size
        self.temporal_feature_size = temporal_feature_size
        self.max_seq_len = max_seq_len
        self.nhead = nhead
        self.nhid = nlayers
        self.dropout = dropout

        encoder_layers_f = nn.TransformerEncoderLayer(
            d_model = frame_feature_size,
            nhead=nhead,
            dim_feedforward=feedforward,
            dropout=dropout
        )
        self.transformer_encoder_f = nn.TransformerEncoder(
            encoder_layers_f, nlayers)

        encoder_layers_t = nn.TransformerEncoderLayer(
            d_model=temporal_feature_size,
            nhead=nhead,
            dim_feedforward=feedforward,
            dropout=dropout
        )
        self.transformer_encoder_t = nn.TransformerEncoder(
            encoder_layers_t, nlayers)

        self.mlp = None

    def forward(self, x, num_frames):
        x = x.permute(2, 0, 1)  # (T, N, C) -> e.g., (300, 64, 1024)

        T, N, C = x.shape[:3]  # (T, N, C)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)
        # print(frame_mask.shape)
        assert C == self.frame_feature_size+self.temporal_feature_size, 'Input should have feature_size {} but got {}.'.format(self.frame_feature_size+self.temporal_feature_size, C)
        # print(x.shape)
        x_f, x_t = x.split(self.frame_feature_size, dim=2)

        output_f = self.transformer_encoder_f(x_f, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)
        output_t = self.transformer_encoder_t(x_t, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)
        # print(output.shape)

        output_f = output_f.permute(1, 0, 2)  # (N, T, C)
        # print(output.shape)
        output_f = output_f * frame_mask.unsqueeze(-1)
        # print(output.shape)
        frame_count = torch.sum(frame_mask, dim=-1, keepdim=True)  # (N, 1)
        # print(frame_count.shape)
        output_f = torch.sum(output_f, dim=-2) / frame_count  # (N, C)
        # print(output.shape)
        # L2 normalize (N, output_dim) IMPORTANT!!!
        output_f = F.normalize(output_f, p=2, dim=1)

        output_t = output_t.permute(1, 0, 2)  # (N, T, C)
        # print(output.shape)
        output_t = output_t * frame_mask.unsqueeze(-1)
        # print(output.shape)
        frame_count = torch.sum(frame_mask, dim=-1, keepdim=True)  # (N, 1)
        # print(frame_count.shape)
        output_t = torch.sum(output_t, dim=-2) / frame_count  # (N, C)
        # print(output.shape)
        # L2 normalize (N, output_dim) IMPORTANT!!!
        output_t = F.normalize(output_t, p=2, dim=1)
        # breakpoint()
        embedding = torch.cat((output_f,output_t),1)
        # breakpoint()
        if self.mlp is not None:
            embedding = self.mlp(embedding)
        return embedding





    def encode(self, x, num_frames):
        x = x.permute(2, 0, 1)  # (T, N, C) -> e.g., (300, 64, 1024)

        T, N, C = x.shape[:3]  # (T, N, C)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
                0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)
        # print(frame_mask.shape)
        assert C == self.frame_feature_size + self.temporal_feature_size, 'Input should have feature_size {} but got {}.'.format(
            self.frame_feature_size + self.temporal_feature_size, C)
        # print(x.shape)
        x_f, x_t = x.split(self.frame_feature_size, dim=2)

        output_f = self.transformer_encoder_f(x_f, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)
        output_t = self.transformer_encoder_t(x_t, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)
        # print(output.shape)

        output_f = output_f.permute(1, 0, 2)  # (N, T, C)
        # print(output.shape)
        output_f = output_f * frame_mask.unsqueeze(-1)
        # print(output.shape)
        frame_count = torch.sum(frame_mask, dim=-1)  # (N, 1)
        # print(frame_count.shape)
        output_f =  torch.narrow(output_f, 1, 0, int(frame_count.item()))
        # print(output.shape)
        # L2 normalize (N, output_dim) IMPORTANT!!!
        output_f = F.normalize(output_f, p=2, dim=2)  # (N, T, C)

        output_t = output_t.permute(1, 0, 2)  # (N, T, C)
        # print(output.shape)
        output_t = output_t * frame_mask.unsqueeze(-1)
        # print(output.shape)
        frame_count = torch.sum(frame_mask, dim=-1)  # (N, 1)
        # print(frame_count.shape)
        output_t = torch.narrow(output_f, 1, 0, int(frame_count.item()))
        # print(output.shape)
        # L2 normalize (N, output_dim) IMPORTANT!!!
        output_t = F.normalize(output_f, p=2, dim=2)  # (N, T, C)

        embedding = torch.cat((output_f, output_t), 2)

        # breakpoint()
        if self.mlp is not None:
            embedding = self.mlp(embedding)
        return embedding
    # output = output.permute(1, 0, 2)  # (N, T, C)
    # output = output * frame_mask.unsqueeze(-1)
    # frame_count = torch.sum(frame_mask, dim=-1)  # (N)
    # output = torch.narrow(output, 1, 0, int(frame_count.item()))
    #
    # # L2 normalize IMPORTANT!!!
    # output = F.normalize(output, p=2, dim=2)  # (N, T, C)
    # if self.mlp is not None:
    #     output = self.mlp(output)
    # return output

class CTCA_LATE_PLUS(nn.Module):
    def __init__(self, frame_feature_size=1024, temporal_feature_size=1024, feedforward=2048, max_seq_len=128, nhead=8, nlayers=1, dropout=0.1):
        super(CTCA_LATE_PLUS, self).__init__()

        self.frame_feature_size = frame_feature_size
        self.temporal_feature_size = temporal_feature_size
        self.max_seq_len = max_seq_len
        self.nhead = nhead
        self.nhid = nlayers
        self.dropout = dropout

        encoder_layers_f = nn.TransformerEncoderLayer(
            d_model = frame_feature_size,
            nhead=nhead,
            dim_feedforward=feedforward,
            dropout=dropout
        )
        self.transformer_encoder_f = nn.TransformerEncoder(
            encoder_layers_f, nlayers)

        encoder_layers_t = nn.TransformerEncoderLayer(
            d_model=temporal_feature_size,
            nhead=nhead,
            dim_feedforward=feedforward,
            dropout=dropout
        )
        self.transformer_encoder_t = nn.TransformerEncoder(
            encoder_layers_t, nlayers)

        self.mlp = None

    def forward(self, x, num_frames):
        x = x.permute(2, 0, 1)  # (T, N, C) -> e.g., (300, 64, 1024)
        # breakpoint()
        T, N, C = x.shape[:3]  # (T, N, C)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)
        # print(frame_mask.shape)
        assert C == self.frame_feature_size+self.temporal_feature_size, 'Input should have feature_size {} but got {}.'.format(self.frame_feature_size+self.temporal_feature_size, C)
        # print(x.shape)
        x_f, x_t = x.split(self.frame_feature_size, dim=2)

        output_f = self.transformer_encoder_f(x_f, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)
        output_t = self.transformer_encoder_t(x_t, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)
        # print(output.shape)

        output_f = output_f.permute(1, 0, 2)  # (N, T, C)
        # print(output_f.shape)
        output_f = output_f * frame_mask.unsqueeze(-1)
        # print(output_f.shape)
        frame_count = torch.sum(frame_mask, dim=-1, keepdim=True)  # (N, 1)
        # print(frame_count.shape)
        output_f = torch.sum(output_f, dim=-2) / frame_count  # (N, C)
        # print(output.shape)
        # L2 normalize (N, output_dim) IMPORTANT!!!
        output_f = F.normalize(output_f, p=2, dim=1)

        output_t = output_t.permute(1, 0, 2)  # (N, T, C)
        # breakpoint()
        # print(output.shape)
        output_t = output_t * frame_mask.unsqueeze(-1)
        # print(output.shape)
        frame_count = torch.sum(frame_mask, dim=-1, keepdim=True)  # (N, 1)
        # print(frame_count.shape)
        output_t = torch.sum(output_t, dim=-2) / frame_count  # (N, C)
        # print(output.shape)
        # L2 normalize (N, output_dim) IMPORTANT!!!
        output_t = F.normalize(output_t, p=2, dim=1)
        # breakpoint()
        embedding = output_f + output_t
        # breakpoint()
        if self.mlp is not None:
            embedding = self.mlp(embedding)
        return embedding


class CTCA_LATE_MAXIMUM(nn.Module):
    def __init__(self, frame_feature_size=1024, temporal_feature_size=1024, feedforward=2048, max_seq_len=128, nhead=8, nlayers=1, dropout=0.1):
        super(CTCA_LATE_MAXIMUM, self).__init__()

        self.frame_feature_size = frame_feature_size
        self.temporal_feature_size = temporal_feature_size
        self.max_seq_len = max_seq_len
        self.nhead = nhead
        self.nhid = nlayers
        self.dropout = dropout

        encoder_layers_f = nn.TransformerEncoderLayer(
            d_model = frame_feature_size,
            nhead=nhead,
            dim_feedforward=feedforward,
            dropout=dropout
        )
        self.transformer_encoder_f = nn.TransformerEncoder(
            encoder_layers_f, nlayers)

        encoder_layers_t = nn.TransformerEncoderLayer(
            d_model=temporal_feature_size,
            nhead=nhead,
            dim_feedforward=feedforward,
            dropout=dropout
        )
        self.transformer_encoder_t = nn.TransformerEncoder(
            encoder_layers_t, nlayers)

        self.mlp = None

    def forward(self, x, num_frames):
        x = x.permute(2, 0, 1)  # (T, N, C) -> e.g., (300, 64, 1024)

        T, N, C = x.shape[:3]  # (T, N, C)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)
        # print(frame_mask.shape)
        assert C == self.frame_feature_size+self.temporal_feature_size, 'Input should have feature_size {} but got {}.'.format(self.frame_feature_size+self.temporal_feature_size, C)
        # print(x.shape)
        x_f, x_t = x.split(self.frame_feature_size, dim=2)

        output_f = self.transformer_encoder_f(x_f, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)
        output_t = self.transformer_encoder_t(x_t, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)
        # print(output.shape)

        output_f = output_f.permute(1, 0, 2)  # (N, T, C)
        # print(output.shape)
        output_f = output_f * frame_mask.unsqueeze(-1)
        # print(output.shape)
        frame_count = torch.sum(frame_mask, dim=-1, keepdim=True)  # (N, 1)
        # print(frame_count.shape)
        output_f = torch.sum(output_f, dim=-2) / frame_count  # (N, C)
        # print(output.shape)
        # L2 normalize (N, output_dim) IMPORTANT!!!
        output_f = F.normalize(output_f, p=2, dim=1)

        output_t = output_t.permute(1, 0, 2)  # (N, T, C)
        # print(output.shape)
        output_t = output_t * frame_mask.unsqueeze(-1)
        # print(output.shape)
        frame_count = torch.sum(frame_mask, dim=-1, keepdim=True)  # (N, 1)
        # print(frame_count.shape)
        output_t = torch.sum(output_t, dim=-2) / frame_count  # (N, C)
        # print(output.shape)
        # L2 normalize (N, output_dim) IMPORTANT!!!
        output_t = F.normalize(output_t, p=2, dim=1)
        # breakpoint()
        embedding = torch.maximum(output_f, output_t)
        # breakpoint()
        if self.mlp is not None:
            embedding = self.mlp(embedding)
        return embedding

class CTCA_LATE_NetVLAD(nn.Module):
    def __init__(self, frame_feature_size=1024, temporal_feature_size=1024, feedforward=2048, max_seq_len=128, nhead=8, nlayers=1, dropout=0.1,netvlad_clusters=16, netvlad_output_dim=512):
        super(CTCA_LATE_NetVLAD, self).__init__()

        self.frame_feature_size = frame_feature_size
        self.temporal_feature_size = temporal_feature_size
        self.max_seq_len = max_seq_len
        self.nhead = nhead
        self.nhid = nlayers
        self.dropout = dropout

        self.netvlad_clusters = netvlad_clusters
        self.netvlad_output_dim = netvlad_output_dim

        self.mlp = None
        self.activation = nn.ReLU()
        encoder_layers_f = nn.TransformerEncoderLayer(
            d_model = frame_feature_size,
            nhead=nhead,
            dim_feedforward=feedforward,
            dropout=dropout
        )
        self.transformer_encoder_f = nn.TransformerEncoder(
            encoder_layers_f, nlayers,  nn.LayerNorm(frame_feature_size))

        encoder_layers_t = nn.TransformerEncoderLayer(
            d_model=temporal_feature_size,
            nhead=nhead,
            dim_feedforward=feedforward,
            dropout=dropout
        )
        self.transformer_encoder_t = nn.TransformerEncoder(
            encoder_layers_t, nlayers,  nn.LayerNorm(temporal_feature_size))
        self.netvlad_f = NetVLAD(self.frame_feature_size, self.netvlad_clusters, outdims=self.netvlad_output_dim)
        self.netvlad_t = NetVLAD(self.temporal_feature_size, self.netvlad_clusters, outdims=self.netvlad_output_dim)



    def forward(self, x, num_frames):
        x = x.permute(2, 0, 1)  # (T, N, C) -> e.g., (300, 64, 1024)

        T, N, C = x.shape[:3]  # (T, N, C)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)
        # print(frame_mask.shape)
        assert C == self.frame_feature_size+self.temporal_feature_size, 'Input should have feature_size {} but got {}.'.format(self.frame_feature_size+self.temporal_feature_size, C)
        # print(x.shape)
        x_f, x_t = x.split(self.frame_feature_size, dim=2)

        output_f = self.transformer_encoder_f(x_f, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)
        output_t = self.transformer_encoder_t(x_t, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)
        # print(output.shape)

        ##  NetVLAD Embedding
        output_f = output_f.permute(1, 0, 2)  # (N, T, C)
        output_f = output_f.unsqueeze(2).permute(0, 3, 1, 2)
        output_f = self.netvlad_f(output_f, mask=frame_mask)

        # L2 normalize (N, output_dim) IMPORTANT!!!
        output_f = F.normalize(output_f, p=2, dim=1)

        ##  NetVLAD Embedding
        output_t = output_t.permute(1, 0, 2)  # (N, T, C)
        output_t = output_t.unsqueeze(2).permute(0, 3, 1, 2)
        output_t = self.netvlad_t(output_t, mask=frame_mask)

        # L2 normalize (N, output_dim) IMPORTANT!!!
        output_t = F.normalize(output_t, p=2, dim=1)

        # breakpoint()
        embedding = torch.cat((output_f,output_t),1)
        # breakpoint()
        if self.mlp is not None:
            embedding = self.mlp(embedding)
        return embedding


class CTCA_LATE_NetVLAD_PLUS(nn.Module):
    def __init__(self, frame_feature_size=1024, temporal_feature_size=1024, feedforward=2048, max_seq_len=128, nhead=8, nlayers=1, dropout=0.1,netvlad_clusters=16, netvlad_output_dim=1024):
        super(CTCA_LATE_NetVLAD_PLUS, self).__init__()

        self.frame_feature_size = frame_feature_size
        self.temporal_feature_size = temporal_feature_size
        self.max_seq_len = max_seq_len
        self.nhead = nhead
        self.nhid = nlayers
        self.dropout = dropout

        self.netvlad_clusters = netvlad_clusters
        self.netvlad_output_dim = netvlad_output_dim

        self.mlp = None
        self.activation = nn.ReLU()
        encoder_layers_f = nn.TransformerEncoderLayer(
            d_model = frame_feature_size,
            nhead=nhead,
            dim_feedforward=feedforward,
            dropout=dropout
        )
        self.transformer_encoder_f = nn.TransformerEncoder(
            encoder_layers_f, nlayers,  nn.LayerNorm(frame_feature_size))

        encoder_layers_t = nn.TransformerEncoderLayer(
            d_model=temporal_feature_size,
            nhead=nhead,
            dim_feedforward=feedforward,
            dropout=dropout
        )
        self.transformer_encoder_t = nn.TransformerEncoder(
            encoder_layers_t, nlayers,  nn.LayerNorm(temporal_feature_size))

        self.netvlad_f = NetVLAD(self.frame_feature_size, self.netvlad_clusters, outdims=self.netvlad_output_dim)
        self.netvlad_t = NetVLAD(self.temporal_feature_size, self.netvlad_clusters, outdims=self.netvlad_output_dim)



    def forward(self, x, num_frames):
        x = x.permute(2, 0, 1)  # (T, N, C) -> e.g., (300, 64, 1024)

        T, N, C = x.shape[:3]  # (T, N, C)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)
        # print(frame_mask.shape)
        assert C == self.frame_feature_size+self.temporal_feature_size, 'Input should have feature_size {} but got {}.'.format(self.frame_feature_size+self.temporal_feature_size, C)
        # print(x.shape)
        x_f, x_t = x.split(self.frame_feature_size, dim=2)

        output_f = self.transformer_encoder_f(x_f, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)
        output_t = self.transformer_encoder_t(x_t, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)
        # print(output.shape)

        ##  NetVLAD Embedding
        output_f = output_f.permute(1, 0, 2)  # (N, T, C)
        output_f = output_f.unsqueeze(2).permute(0, 3, 1, 2)
        output_f = self.netvlad_f(output_f, mask=frame_mask)

        # L2 normalize (N, output_dim) IMPORTANT!!!
        output_f = F.normalize(output_f, p=2, dim=1)

        ##  NetVLAD Embedding
        output_t = output_t.permute(1, 0, 2)  # (N, T, C)
        output_t = output_t.unsqueeze(2).permute(0, 3, 1, 2)
        output_t = self.netvlad_t(output_t, mask=frame_mask)

        # L2 normalize (N, output_dim) IMPORTANT!!!
        output_t = F.normalize(output_t, p=2, dim=1)

        # breakpoint()
        embedding = output_f+output_t
        # breakpoint()
        if self.mlp is not None:
            embedding = self.mlp(embedding)
        return embedding

class CTCA_LATE_FC(nn.Module):
    def __init__(self, frame_feature_size=1024, temporal_feature_size=1024, feedforward=2048, max_seq_len=128, nhead=8, nlayers=1, dropout=0.1):
        super(CTCA_LATE_FC, self).__init__()

        self.frame_feature_size = frame_feature_size
        self.temporal_feature_size = temporal_feature_size
        self.max_seq_len = max_seq_len
        self.nhead = nhead
        self.nhid = nlayers
        self.dropout = dropout
        self.linear_f = nn.Linear(frame_feature_size,512, bias=False)
        self.linear_t = nn.Linear(temporal_feature_size,512, bias=False)
        self.norm_f = nn.LayerNorm(512)
        self.norm_t = nn.LayerNorm(512)
        self.mlp = None
        self.activation = nn.ReLU()
        encoder_layers_f = nn.TransformerEncoderLayer(
            d_model = frame_feature_size,
            nhead=nhead,
            dim_feedforward=feedforward,
            dropout=dropout
        )
        self.transformer_encoder_f = nn.TransformerEncoder(
            encoder_layers_f, nlayers, nn.Sequential(self.linear_f, self.activation, self.norm_f))


        encoder_layers_t = nn.TransformerEncoderLayer(
            d_model=temporal_feature_size,
            nhead=nhead,
            dim_feedforward=feedforward,
            dropout=dropout
        )
        self.transformer_encoder_t = nn.TransformerEncoder(
            encoder_layers_t, nlayers, nn.Sequential(self.linear_t, self.activation, self.norm_t))


        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear_f.weight, std=1 / math.sqrt(512))
        nn.init.normal_(self.linear_t.weight, std=1 / math.sqrt(512))


    def forward(self, x, num_frames):
        x = x.permute(2, 0, 1)  # (T, N, C) -> e.g., (300, 64, 1024)

        T, N, C = x.shape[:3]  # (T, N, C)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)
        # print(frame_mask.shape)
        assert C == self.frame_feature_size+self.temporal_feature_size, 'Input should have feature_size {} but got {}.'.format(self.frame_feature_size+self.temporal_feature_size, C)
        # print(x.shape)
        x_f, x_t = x.split(self.frame_feature_size, dim=2)

        output_f = self.transformer_encoder_f(x_f, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)
        output_t = self.transformer_encoder_t(x_t, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)
        # print(output.shape)

        output_f = output_f.permute(1, 0, 2)  # (N, T, C)
        # print(output.shape)
        output_f = output_f * frame_mask.unsqueeze(-1)
        # print(output.shape)
        frame_count = torch.sum(frame_mask, dim=-1, keepdim=True)  # (N, 1)
        # print(frame_count.shape)
        output_f = torch.sum(output_f, dim=-2) / frame_count  # (N, C)
        # print(output.shape)
        # L2 normalize (N, output_dim) IMPORTANT!!!
        output_f = F.normalize(output_f, p=2, dim=1)

        output_t = output_t.permute(1, 0, 2)  # (N, T, C)
        # print(output.shape)
        output_t = output_t * frame_mask.unsqueeze(-1)
        # print(output.shape)
        frame_count = torch.sum(frame_mask, dim=-1, keepdim=True)  # (N, 1)
        # print(frame_count.shape)
        output_t = torch.sum(output_t, dim=-2) / frame_count  # (N, C)
        # print(output.shape)
        # L2 normalize (N, output_dim) IMPORTANT!!!
        output_t = F.normalize(output_t, p=2, dim=1)
        # breakpoint()
        embedding = torch.cat((output_f,output_t),1)
        # breakpoint()
        if self.mlp is not None:
            embedding = self.mlp(embedding)
        return embedding


class CTCA_LATE_FC_BIAS(nn.Module):
    def __init__(self, frame_feature_size=1024, temporal_feature_size=1024, feedforward=2048, max_seq_len=128, nhead=8, nlayers=1, dropout=0.1):
        super(CTCA_LATE_FC_BIAS, self).__init__()

        self.frame_feature_size = frame_feature_size
        self.temporal_feature_size = temporal_feature_size
        self.max_seq_len = max_seq_len
        self.nhead = nhead
        self.nhid = nlayers
        self.dropout = dropout
        self.linear_f = nn.Linear(frame_feature_size,512, bias=True)
        self.linear_t = nn.Linear(temporal_feature_size,512, bias=True)
        self.norm_f = nn.LayerNorm(512)
        self.norm_t = nn.LayerNorm(512)
        self.mlp = None
        self.activation = nn.ReLU()
        encoder_layers_f = nn.TransformerEncoderLayer(
            d_model = frame_feature_size,
            nhead=nhead,
            dim_feedforward=feedforward,
            dropout=dropout
        )
        self.transformer_encoder_f = nn.TransformerEncoder(
            encoder_layers_f, nlayers, nn.Sequential(self.linear_f, self.activation, self.norm_f))


        encoder_layers_t = nn.TransformerEncoderLayer(
            d_model=temporal_feature_size,
            nhead=nhead,
            dim_feedforward=feedforward,
            dropout=dropout
        )
        self.transformer_encoder_t = nn.TransformerEncoder(
            encoder_layers_t, nlayers, nn.Sequential(self.linear_t, self.activation, self.norm_t))


        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear_f.weight, std=1 / math.sqrt(512))
        nn.init.normal_(self.linear_t.weight, std=1 / math.sqrt(512))


    def forward(self, x, num_frames):
        x = x.permute(2, 0, 1)  # (T, N, C) -> e.g., (300, 64, 1024)

        T, N, C = x.shape[:3]  # (T, N, C)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)
        # print(frame_mask.shape)
        assert C == self.frame_feature_size+self.temporal_feature_size, 'Input should have feature_size {} but got {}.'.format(self.frame_feature_size+self.temporal_feature_size, C)
        # print(x.shape)
        x_f, x_t = x.split(self.frame_feature_size, dim=2)

        output_f = self.transformer_encoder_f(x_f, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)
        output_t = self.transformer_encoder_t(x_t, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)
        # print(output.shape)

        output_f = output_f.permute(1, 0, 2)  # (N, T, C)
        # print(output.shape)
        output_f = output_f * frame_mask.unsqueeze(-1)
        # print(output.shape)
        frame_count = torch.sum(frame_mask, dim=-1, keepdim=True)  # (N, 1)
        # print(frame_count.shape)
        output_f = torch.sum(output_f, dim=-2) / frame_count  # (N, C)
        # print(output.shape)
        # L2 normalize (N, output_dim) IMPORTANT!!!
        output_f = F.normalize(output_f, p=2, dim=1)

        output_t = output_t.permute(1, 0, 2)  # (N, T, C)
        # print(output.shape)
        output_t = output_t * frame_mask.unsqueeze(-1)
        # print(output.shape)
        frame_count = torch.sum(frame_mask, dim=-1, keepdim=True)  # (N, 1)
        # print(frame_count.shape)
        output_t = torch.sum(output_t, dim=-2) / frame_count  # (N, C)
        # print(output.shape)
        # L2 normalize (N, output_dim) IMPORTANT!!!
        output_t = F.normalize(output_t, p=2, dim=1)
        # breakpoint()
        embedding = torch.cat((output_f,output_t),1)
        # breakpoint()
        if self.mlp is not None:
            embedding = self.mlp(embedding)
        return embedding

class CTCA_LATE_MAX(nn.Module):
    def __init__(self, frame_feature_size=1024, temporal_feature_size=1024, feedforward=2048, max_seq_len=128, nhead=8, nlayers=1, dropout=0.1):
        super(CTCA_LATE_MAX, self).__init__()

        self.frame_feature_size = frame_feature_size
        self.temporal_feature_size = temporal_feature_size
        self.max_seq_len = max_seq_len
        self.nhead = nhead
        self.nhid = nlayers
        self.dropout = dropout
        self.pool = nn.MaxPool1d(2, stride=2)

        encoder_layers_f = nn.TransformerEncoderLayer(
            d_model = frame_feature_size,
            nhead=nhead,
            dim_feedforward=feedforward,
            dropout=dropout
        )
        self.transformer_encoder_f = nn.TransformerEncoder(
            encoder_layers_f, nlayers)

        encoder_layers_t = nn.TransformerEncoderLayer(
            d_model=temporal_feature_size,
            nhead=nhead,
            dim_feedforward=feedforward,
            dropout=dropout
        )
        self.transformer_encoder_t = nn.TransformerEncoder(
            encoder_layers_t, nlayers)

        self.mlp = None

    def forward(self, x, num_frames):
        x = x.permute(2, 0, 1)  # (T, N, C) -> e.g., (300, 64, 1024)

        T, N, C = x.shape[:3]  # (T, N, C)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)
        # print(frame_mask.shape)
        assert C == self.frame_feature_size+self.temporal_feature_size, 'Input should have feature_size {} but got {}.'.format(self.frame_feature_size+self.temporal_feature_size, C)
        # print(x.shape)
        x_f, x_t = x.split(self.frame_feature_size, dim=2)

        output_f = self.transformer_encoder_f(x_f, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)
        output_t = self.transformer_encoder_t(x_t, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)
        # print(output.shape)

        output_f = output_f.permute(1, 0, 2)  # (N, T, C)
        # print(output.shape)
        output_f = output_f * frame_mask.unsqueeze(-1)
        # print(output.shape)
        frame_count = torch.sum(frame_mask, dim=-1, keepdim=True)  # (N, 1)
        # print(frame_count.shape)
        output_f = torch.sum(output_f, dim=-2) / frame_count  # (N, C)
        # print(output.shape)
        # L2 normalize (N, output_dim) IMPORTANT!!!
        output_f = F.normalize(output_f, p=2, dim=1)

        # Pooling ( N, output_dim /2)
        output_f = torch.unsqueeze(output_f, 1)
        output_f = self.pool(output_f)
        output_f = torch.squeeze(output_f,1)


        output_t = output_t.permute(1, 0, 2)  # (N, T, C)
        # print(output.shape)
        output_t = output_t * frame_mask.unsqueeze(-1)
        # print(output.shape)
        frame_count = torch.sum(frame_mask, dim=-1, keepdim=True)  # (N, 1)
        # print(frame_count.shape)
        output_t = torch.sum(output_t, dim=-2) / frame_count  # (N, C)

        # print(output.shape)
        # L2 normalize (N, output_dim) IMPORTANT!!!
        output_t = F.normalize(output_t, p=2, dim=1)
        # Pooling ( N, output_dim /2)
        output_t = torch.unsqueeze(output_t, 1)
        output_t = self.pool(output_t)
        output_t = torch.squeeze(output_t,1)
        # Pooling ( N, output_dim /2)

        # breakpoint()
        embedding = torch.cat((output_f,output_t),1)

        # breakpoint()
        if self.mlp is not None:
            embedding = self.mlp(embedding)
        return embedding

class CTCA_NetVLAD_AUX(nn.Module):
    def __init__(self, feature_size=2048, max_seq_len=128, nhead=8, nlayers=1, dropout=0.1, netvlad_clusters=64, netvlad_output_dim=1024):
        super(CTCA_NetVLAD_AUX, self).__init__()

        self.feature_size = feature_size
        self.max_seq_len = max_seq_len
        self.nhead = nhead
        self.nhid = nlayers
        self.dropout = dropout
        self.netvlad_clusters = netvlad_clusters
        self.netvlad_output_dim = netvlad_output_dim

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=feature_size,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, nlayers,nn.LayerNorm(self.feature_size))

        self.netvlad = NetVLAD(self.feature_size, self.netvlad_clusters, outdims=self.netvlad_output_dim)

    def forward(self, x, num_frames):
        x = x.permute(2, 0, 1)  # (T, N, C) -> e.g., (300, 64, 1024)

        T, N, C = x.shape[:3]  # (T, N, C)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)
        # print(frame_mask.shape)
        assert C == self.feature_size, 'Input should have feature_size {} but got {}.'.format(self.feature_size, C)
        # print(x.shape)
        output = self.transformer_encoder(
            x, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)
        # print(output.shape)
        output = output.permute(1, 0, 2)  # (N, T, C)

       ##  NetVLAD Embedding

        output = output.unsqueeze(2).permute(0, 3, 1, 2)
        output = self.netvlad(output, mask=frame_mask)

        # L2 normalize (N, output_dim) IMPORTANT!!!
        v_embedding = F.normalize(output, p=2, dim=1)

        return v_embedding

    def encode(self, x, num_frames):
        x = x.permute(2, 0, 1)  # (T, N, C) -> e.g., (300, 64, 1024)

        T, N, C = x.shape[:3]  # (T, N, C)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)
        # print(frame_mask.shape)
        assert C == self.feature_size, 'Input should have feature_size {} but got {}.'.format(self.feature_size, C)
        # print(x.shape)
        output = self.transformer_encoder(
            x, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)
        # print(output.shape)
        output = output.permute(1, 0, 2)  # (N, T, C)


        ## Transformer Embedding
        # print(output.shape)
        t_out = output * frame_mask.unsqueeze(-1)
        # print(output.shape)
        frame_count = torch.sum(frame_mask, dim=-1, keepdim=True)  # (N, 1)
        # print(frame_count.shape)
        t_out = torch.sum(t_out, dim=-2) / frame_count  # (N, C)
        # print(output.shape)
        # L2 normalize (N, output_dim) IMPORTANT!!!
        t_embedding = F.normalize(t_out, p=2, dim=1)

       ##  NetVLAD Embedding

        output = output.unsqueeze(2).permute(0, 3, 1, 2)
        output = self.netvlad(output, mask=frame_mask)

        # L2 normalize (N, output_dim) IMPORTANT!!!
        v_embedding = F.normalize(output, p=2, dim=1)

        return t_embedding, v_embedding


class CTCA_NetVLAD(nn.Module):
    def __init__(self, feature_size=2048, feedforward=2048, max_seq_len=128, nhead=8, nlayers=1, dropout=0.1, netvlad_clusters=64, netvlad_output_dim=1024):
        super(CTCA_NetVLAD, self).__init__()

        self.feature_size = feature_size
        self.max_seq_len = max_seq_len
        self.nhead = nhead
        self.nhid = nlayers
        self.dropout = dropout
        self.netvlad_clusters = netvlad_clusters
        self.netvlad_output_dim = netvlad_output_dim
        self.feedforward = feedforward
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=feature_size,
            nhead=nhead,
            dim_feedforward=self.feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, nlayers,nn.LayerNorm(self.feature_size))

        self.netvlad = NetVLAD(self.feature_size, self.netvlad_clusters, outdims=self.netvlad_output_dim)


    def forward(self, x, num_frames):
        x = x.permute(2, 0, 1)  # (T, N, C) -> e.g., (300, 64, 1024)

        T, N, C = x.shape[:3]  # (T, N, C)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)
        # print(frame_mask.shape)
        assert C == self.feature_size, 'Input should have feature_size {} but got {}.'.format(self.feature_size, C)
        # print(x.shape)
        output = self.transformer_encoder(
            x, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)
        # print(output.shape)
        output = output.permute(1, 0, 2)  # (N, T, C)
        # print(output.shape)

        output = output.unsqueeze(2).permute(0, 3, 1, 2)  # (N, C, T, 1)
        # breakpoint()
        output = self.netvlad(output, mask=frame_mask) # (N, out_dims)

        # L2 normalize (N, output_dim) IMPORTANT!!!
        embedding = F.normalize(output, p=2, dim=1)

        return embedding




class TCA(nn.Module):
    def __init__(self, feature_size=1024, max_seq_len=128, nhead=8, nlayers=1, dropout=0.1):
        super(TCA, self).__init__()

        self.feature_size = feature_size
        self.max_seq_len = max_seq_len
        self.nhead = nhead
        self.nhid = nlayers
        self.dropout = dropout

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=feature_size,
            nhead=nhead,
            dim_feedforward=4096,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, nlayers)
        self.mlp = None

    def forward(self, x, num_frames):
        x = x.permute(2, 0, 1)  # (T, N, C) -> e.g., (300, 64, 1024)

        T, N, C = x.shape[:3]  # (T, N, C)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)

        assert C == self.feature_size, 'Input should have feature_size {} but got {}.'.format(self.feature_size, C)

        output = self.transformer_encoder(
            x, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)

        output = output.permute(1, 0, 2)  # (N, T, C)
        # print(output.shape)
        output = output * frame_mask.unsqueeze(-1)
        # print(output.shape)
        frame_count = torch.sum(frame_mask, dim=-1, keepdim=True)  # (N, 1)
        # print(frame_count.shape)
        output = torch.sum(output, dim=-2) / frame_count  # (N, C)
        # print(output.shape)
        # L2 normalize (N, output_dim) IMPORTANT!!!
        embedding = F.normalize(output, p=2, dim=1)
        if self.mlp is not None:
            embedding = self.mlp(embedding)
        # breakpoint()
        return embedding

    def encode(self, x, num_frames):
        x = x.permute(2, 0, 1)  # (T, N, C)

        T, N, C = x.shape[:3]  # (T, N, C)
        # mask padded frame feature
        if len(num_frames.shape) == 1:
            num_frames = num_frames.unsqueeze(1)
        frame_mask = (
            0 < num_frames - torch.arange(0, T).cuda()
        ).float()  # (N, T)
        assert C == self.feature_size, 'Input should have feature_size {} but got {}.'.format(self.feature_size, C)

        output = self.transformer_encoder(
            x, src_key_padding_mask=(1 - frame_mask).bool())  # (T, N, C)
        output = output.permute(1, 0, 2)  # (N, T, C)
        output = output * frame_mask.unsqueeze(-1)
        frame_count = torch.sum(frame_mask, dim=-1)  # (N)
        output = torch.narrow(output, 1, 0, int(frame_count.item()))

        # L2 normalize IMPORTANT!!!
        output = F.normalize(output, p=2, dim=2)  # (N, T, C)
        if self.mlp is not None:
            output = self.mlp(output)
        return output


class simple_MLP(nn.Module):
    def __init__(self, dims):
        super(simple_MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2])
        )

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=1024, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 1024)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        self.encoder_q = base_encoder
        self.encoder_k = copy.deepcopy(self.encoder_q)
        self.is_mlp = mlp
        if self.is_mlp:  # hack: brute-force replacement
            self.encoder_q.mlp = simple_MLP([1024,1024,1024])
            self.encoder_k.mlp = simple_MLP([1024,1024,1024])
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, a, p, n, len_a, len_p, len_n):
        """
        Input:
            a: a batch of anchor logits
            p: a batch of positive logits
            n: a bigger batch of negative logits
        Output:
            logits, targets
        """

        if len(n.size()) > 3:
            n = n.view(-1, n.size()[2], n.size()[3])
            len_n = len_n.view(-1, 1)

        # compute query features
        q = self.encoder_q(a, len_a)  # queries: NxC
        q = F.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            p = self.encoder_k(p, len_p)  # anchors: NxC
            p = F.normalize(p, dim=1)
            k = self.encoder_k(n, len_n)  # keys: kNxC
            k = F.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, p]).unsqueeze(-1)
        # negative logits: NxK
        # breakpoint()
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        # print(p.shape,q.shape,self.queue.clone().shape)
        # print(l_pos.shape, l_neg.shape)
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

class MoCoAUX(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, t_dim=1024, v_dim=1024, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 1024)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCoAUX, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        self.encoder_q = base_encoder
        self.encoder_k = copy.deepcopy(self.encoder_q)
        self.is_mlp = mlp
        if self.is_mlp:  # hack: brute-force replacement
            pass
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue_t", torch.randn(t_dim, K))
        self.register_buffer("queue_v", torch.randn(v_dim, K))
        self.queue_t = F.normalize(self.queue_t, dim=0)
        self.queue_v = F.normalize(self.queue_v, dim=0)

        self.register_buffer("queue_ptr_t", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_ptr_v", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue_t(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr_t)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_t[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr_t[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_v(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr_v)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_v[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr_v[0] = ptr

    def forward(self, a, p, n, len_a, len_p, len_n):
        """
        Input:
            a: a batch of anchor logits
            p: a batch of positive logits
            n: a bigger batch of negative logits
        Output:
            logits, targets
        """

        if len(n.size()) > 3:
            n = n.view(-1, n.size()[2], n.size()[3])
            len_n = len_n.view(-1, 1)

        # compute query features
        q_t, q_v = self.encoder_q.encode(a, len_a)  # queries: NxC
        q_t = F.normalize(q_t, dim=1)
        q_v = F.normalize(q_v, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            p_t,p_v = self.encoder_k.encode(p, len_p)  # anchors: NxC
            p_t = F.normalize(p_t, dim=1)
            p_v = F.normalize(p_v, dim=1)

            k_t,k_v = self.encoder_k.encode(n, len_n)  # keys: kNxC
            k_t = F.normalize(k_t, dim=1)
            k_v = F.normalize(k_v, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos_t = torch.einsum('nc,nc->n', [q_t, p_t]).unsqueeze(-1)
        l_pos_v = torch.einsum('mc,mc->m', [q_v, p_v]).unsqueeze(-1)
        # negative logits: NxK

        l_neg_t = torch.einsum('nc,ck->nk', [q_t, self.queue_t.clone().detach()])
        l_neg_v = torch.einsum('md,dl->ml', [q_v, self.queue_v.clone().detach()])
        # print(p.shape,q.shape,self.queue.clone().shape)
        # print(l_pos.shape, l_neg.shape)
        # logits: Nx(1+K)
        logits_t = torch.cat([l_pos_t, l_neg_t], dim=1)
        logits_v = torch.cat([l_pos_v, l_neg_v], dim=1)

        # apply temperature
        logits_t /= self.T
        logits_v /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits_t.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue_t(k_t)
        self._dequeue_and_enqueue_v(k_v)

        return logits_t, logits_v, labels

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    return hvd.allgather(tensor.contiguous())



