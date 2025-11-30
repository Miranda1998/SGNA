import collections

import torch
import torch.nn.functional as F
from torch import nn


class ReLUNetworkPerScenario(nn.Module):
    """
        Multilayer neural network.  
    """

    def __init__(self, feature_dim, hidden_dims, dropout=0):
        """
            Builds a neural network from a list of hidden dimensions.  
            If the list is empty, then the model is simply linear regression. 
        """
        super(ReLUNetworkPerScenario, self).__init__()

        self.input_dim = feature_dim
        self.hidden_dims = hidden_dims
        self.output_dim = 1
        # self.dropout = dropout

        self.downsample_layer = nn.Linear(self.input_dim - 10, 10)

        self.layers = nn.Sequential()

        self.layers.append(nn.Linear(self.input_dim, self.hidden_dims[0]))
        self.layers.append(nn.ReLU())

        for i in range(len(self.hidden_dims) - 1):
            self.layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
            self.layers.append(nn.ReLU())
            if dropout:
                self.layers.append(nn.Dropout(dropout))

        self.layers.append(nn.Linear(self.hidden_dims[-1], self.output_dim))

        # self.layers = collections.OrderedDict()
        #
        # if len(self.hidden_dims) == 0:
        #     self.layers["layer_0"] = nn.Linear(self.input_dim, self.output_dim)
        #
        # else:  # build layers from list
        #     print("Building NN with hidden dims: ", self.hidden_dims)
        #     print("Input dim: ", self.input_dim)
        #     self.layers["layer_in"] = nn.Linear(self.input_dim, self.hidden_dims[0])
        #     self.layers["activation_in"] = nn.ReLU()
        #     if self.dropout:
        #         self.layers["dropout_in"] = nn.Dropout(self.dropout)
        #
        #     for i in range(len(self.hidden_dims) - 1):
        #         self.layers[f"layer_{i}"] = nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1])
        #         self.layers[f"activation_{i}"] = nn.ReLU()
        #         if self.dropout:
        #             self.layers[f"dropout_{i}"] = nn.Dropout(self.dropout)
        #
        #     self.layers[f"layer_out"] = nn.Linear(self.hidden_dims[-1], self.output_dim)
        #
        # self.layers = torch.nn.Sequential(self.layers)

    def forward(self, x):
        """ Forward pass. """
        # print("x.shape", x.shape)
        x2 = self.custom_min_max_normalize(x[:, 10:])  # 对后720维进行归一化
        x = torch.cat((x[:, :10], x2), dim=-1)

        # x2 = self.downsample_layer(x[:, 10:])
        # x = torch.cat((x[:, :10], x2), dim=-1)

        x = self.layers(x)

        return x

    def custom_min_max_normalize(self, x):
        """
        对x进行归一化，后720维按照指定的范围进行Min-Max归一化。
        - 如果值在27到31之间，按照[27, 31]归一化
        - 如果值在-96到-90之间，按照[-96, -90]归一化
        """
        # 定义归一化的范围
        range_27_31 = (27, 31)
        range_neg96_neg90 = (-96, -90)

        # 创建一个新的 x，保持原有数据结构
        normalized_x = x.clone()

        # 计算 27 到 31 范围内的掩码
        mask_27_31 = (x >= range_27_31[0]) & (x <= range_27_31[1])

        # 计算 -96 到 -90 范围内的掩码
        mask_neg96_neg90 = (x >= range_neg96_neg90[0]) & (x <= range_neg96_neg90[1])

        # 对于在 27 到 31 范围内的值进行归一化
        normalized_x[mask_27_31] = (x[mask_27_31] - range_27_31[0]) / (range_27_31[1] - range_27_31[0])

        # 对于在 -96 到 -90 范围内的值进行归一化
        normalized_x[mask_neg96_neg90] = (x[mask_neg96_neg90] - range_neg96_neg90[0]) / (
                    range_neg96_neg90[1] - range_neg96_neg90[0])

        # 对于不在这两个范围内的值，可以保持原值，或者你也可以选择其他方式处理
        # normalized_x[~mask_27_31 & ~mask_neg96_neg90] = x[~mask_27_31 & ~mask_neg96_neg90]  # 保持原值

        return normalized_x

    # def custom_min_max_normalize(self, x):
    #     """
    #     对x进行归一化，后720维按照指定的范围进行Min-Max归一化。
    #     - 如果值在27到31之间，按照[27, 31]归一化
    #     - 如果值在-96到-90之间，按照[-96, -90]归一化
    #     """
    #     # 获取x的形状，假设x的shape是 (batch_size, 720)
    #     batch_size, _ = x.shape
    #
    #     # 创建一个新的x，保持原有数据结构
    #     normalized_x = x.clone()
    #
    #     # 定义归一化的范围
    #     range_27_31 = (27, 31)
    #     range_neg96_neg90 = (-96, -90)
    #
    #     # 归一化逻辑：针对后720维
    #     for i in range(batch_size):
    #         for j in range(720):  # 逐个遍历后720维
    #             value = x[i, j]
    #
    #             if range_27_31[0] <= value <= range_27_31[1]:
    #                 # 如果值在27到31之间，进行归一化
    #                 normalized_x[i, j] = (value - range_27_31[0]) / (range_27_31[1] - range_27_31[0])
    #             elif range_neg96_neg90[0] <= value <= range_neg96_neg90[1]:
    #                 # 如果值在-96到-90之间，进行归一化
    #                 normalized_x[i, j] = (value - range_neg96_neg90[0]) / (range_neg96_neg90[1] - range_neg96_neg90[0])
    #             else:
    #                 # 如果不在这两个范围内，可以选择不做任何变化或按其他规则处理
    #                 # 这里选择不做处理，保持原值
    #                 print('Value out of range:', value.item())
    #                 normalized_x[i, j] = value
    #
    #     return normalized_x


class ReLUNetworkExpected(nn.Module):
    """Multilayer neural network.
    """

    def __init__(self,
                 fs_input_dim,
                 ss_input_dim,
                 ss_hidden_dim,
                 ss_embed_dim1,
                 ss_embed_dim2,
                 relu_hidden_dim,
                 dropout=0,
                 agg_type="mean",
                 bias=False):
        """
            Builds a neural network from a list of hidden dimensions.  
            If the list is empty, then the model is simply linear regression. 
        """
        super(ReLUNetworkExpected, self).__init__()

        self.fs_input_dim = fs_input_dim

        self.ss_input_dim = ss_input_dim
        self.ss_hidden_dim = ss_hidden_dim
        self.ss_embed_dim1 = ss_embed_dim1
        self.ss_embed_dim2 = ss_embed_dim2

        self.relu_hidden_dim = relu_hidden_dim

        self.dropout = dropout
        self.bias = bias
        self.agg_type = agg_type

        self.output_dim = 1

        # layers for scenario input
        self.scen_input = nn.Linear(self.ss_input_dim, self.ss_hidden_dim, bias=self.bias)
        self.scen_embed1 = nn.Linear(self.ss_hidden_dim, self.ss_embed_dim1, bias=self.bias)
        self.scen_embed2 = nn.Linear(self.ss_embed_dim1, self.ss_embed_dim2)

        # for relu layer
        self.relu_input = nn.Linear(self.fs_input_dim + self.ss_embed_dim2, self.relu_hidden_dim)
        self.relu_output = nn.Linear(self.relu_hidden_dim, self.output_dim)

    def forward(self, x_fs, x_scen, x_n_scen=None):
        """ Forward pass. """

        # embed scenarios
        x_scen_embed = self.embed_scenarios(x_scen, x_n_scen)

        # concat first stage solution and scenario embedding
        x = torch.cat((x_fs, x_scen_embed), 1)

        # get aggregate prediction
        x = self.relu_input(x)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, p=self.dropout)

        x = self.relu_output(x)

        return x

    def embed_scenarios(self, x_scen, x_n_scen=None):
        """ Get scenario embedding
        """
        # for each batch, pass-non padded values in. 
        if x_n_scen is not None:
            x_scen_embed = []
            for i in range(x_scen.shape[0]):
                n_scen = int(x_n_scen[i].item())
                x_scen_in = x_scen[i, :n_scen]
                x_scen_in = torch.reshape(x_scen_in, (1, x_scen_in.shape[0], x_scen_in.shape[1]))

                # embed 
                x_scen_embed_i = self.scen_input(x_scen_in)
                x_scen_embed_i = F.relu(x_scen_embed_i)
                if self.dropout:
                    x_scen_embed_i = F.dropout(x_scen_embed_i, p=self.dropout)

                x_scen_embed_i = self.scen_embed1(x_scen_embed_i)
                x_scen_embed_i = F.relu(x_scen_embed_i)
                if self.dropout:
                    x_scen_embed_i = F.dropout(x_scen_embed_i, p=self.dropout)

                if self.agg_type == "sum":
                    x_scen_embed_i = torch.sum(x_scen_embed_i, axis=1)  # sum all inputs
                elif self.agg_type == "mean":
                    x_scen_embed_i = torch.mean(x_scen_embed_i, axis=1)  # mean all inputs

                x_scen_embed_i = self.scen_embed2(x_scen_embed_i)
                x_scen_embed_i = F.relu(x_scen_embed_i)
                if self.dropout:
                    x_scen_embed_i = F.dropout(x_scen_embed_i, p=self.dropout)

                x_scen_embed.append(x_scen_embed_i)

            x_scen_embed = torch.stack(x_scen_embed)
            x_scen_embed = torch.reshape(x_scen_embed, (x_scen_embed.shape[0], x_scen_embed.shape[2]))
            if self.dropout:
                x_scen_embed = F.dropout(x_scen_embed, p=self.dropout)

        # assume no padding, i.e. full scenario set
        else:
            x_scen_embed = self.scen_input(x_scen)
            x_scen_embed = F.relu(x_scen_embed)
            if self.dropout:
                x_scen_embed = F.dropout(x_scen_embed, p=self.dropout)

            x_scen_embed = self.scen_embed1(x_scen_embed)
            x_scen_embed = F.relu(x_scen_embed)
            if self.dropout:
                x_scen_embed = F.dropout(x_scen_embed, p=self.dropout)

            if self.agg_type == "sum":
                x_scen_embed = torch.sum(x_scen_embed, axis=1)  # sum all inputs
            elif self.agg_type == "mean":
                x_scen_embed = torch.mean(x_scen_embed, axis=1)  # mean all inputs

            x_scen_embed = self.scen_embed2(x_scen_embed)
            x_scen_embed = F.relu(x_scen_embed)
            if self.dropout:
                x_scen_embed = F.dropout(x_scen_embed, p=self.dropout)

        return x_scen_embed
