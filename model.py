import logging
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse
import torch.nn.functional as F
import yaml

def get_positional_encoding(pos_vector, device):
    # sin on x, cos on y
    batch_size, dim = pos_vector.shape
    positional_encoding = torch.zeros(batch_size, dim, device=device)
    
    i = torch.arange(dim // 2, device=device, dtype=torch.float32)  # i를 GPU에서 생성
    div_term = torch.exp(i * (-torch.log(torch.tensor(10000.0, device=device)) / (dim // 2)))  # torch.log 사용

    positional_encoding[:, 0::2] = torch.sin(pos_vector[:, 0::2] / div_term)
    positional_encoding[:, 1::2] = torch.cos(pos_vector[:, 1::2] / div_term)

    return positional_encoding

class GraphPool(nn.Module):
    def __init__(self, in_dim: int, pooling_ratio: float = 0.5):
        super().__init__()
        self.pooling_ratio = pooling_ratio
        self.score_layer = nn.Linear(in_dim, 1)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        """
        Graph pooling: 노드의 중요도를 계산하여 top-k 노드만 선택
        
        Args:
            x: 입력 텐서 (batch_size, num_nodes, feature_dim)
            
        Returns:
            pooled_x: 풀링된 텐서 (batch_size, k_nodes, feature_dim)
        """
        batch_size, num_nodes, feature_dim = x.size()
        
        # Dropout 적용
        x_dropout = self.dropout(x)
        
        # 각 노드의 score 계산
        scores = self.score_layer(x_dropout)
        scores = torch.sigmoid(scores)  # (batch_size, num_nodes, 1)
        
        # Top-k 노드 선택
        k_nodes = max(int(num_nodes * self.pooling_ratio), 1)
        _, top_indices = torch.topk(scores, k_nodes, dim=1)
        
        # 선택된 노드들의 인덱스를 feature dimension으로 확장
        top_indices = top_indices.expand(-1, -1, feature_dim)
        
        # Score를 적용하고 top-k 노드 선택
        x = x * scores
        pooled_x = torch.gather(x, dim=1, index=top_indices)
        
        return pooled_x

class LandmarkGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim1=32, hidden_dim2=16, out_dim=16, heads=8, dropout=0.2):
        super(LandmarkGAT, self).__init__()

        # 각 레이어의 차원을 점진적으로 축소
        self.gat1 = GATConv(in_dim, hidden_dim1, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim1 * heads, hidden_dim2, heads=heads, concat=True)
        self.gat3 = GATConv(hidden_dim2 * heads, out_dim, heads=1, concat=False)

        self.bn1 = nn.BatchNorm1d(hidden_dim1 * heads)
        self.bn2 = nn.BatchNorm1d(hidden_dim2 * heads)

        self.dropout = nn.Dropout(dropout)
        self.pool = GraphPool(out_dim, pooling_ratio=0.5)

    def forward(self, x, edge_index):
        batch_size, num_nodes, hidden_dim = x.shape
        x = x.view(-1, hidden_dim)

        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = nn.SELU()(x)
        x = self.dropout(x)

        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = nn.SELU()(x)
        x = self.dropout(x)

        x = self.gat3(x, edge_index)
        x = x.view(batch_size, num_nodes, -1)
        x = self.pool(x)

        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1 * 1 convolution for skip connection
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = None

    def forward(self, x):
        residual = self.skip_conv(x) if self.skip_conv is not None else x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual

        return out

class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x1, x2, master=None):
        '''
        x1  :(#bs, #node, #dim)
        x2  :(#bs, #node, #dim)
        '''
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)

        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)

        x = torch.cat([x1, x2], dim=1)

        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)

        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x, num_type1, num_type2)

        # directional edge for master node
        master = self._update_master(x, master)

        # projection
        x = self._project(x, att_map)

        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)

        x1 = x.narrow(1, 0, num_type1)
        x2 = x.narrow(1, num_type1, num_type2)

        return x1, x2, master

    def _update_master(self, x, master):

        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)

        return master

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map_master(self, x, master):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))

        att_map = torch.matmul(att_map, self.att_weightM)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _derive_att_map(self, x, num_type1, num_type2):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)

        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)

        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11)
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22)
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12)
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12)

        att_map = att_board

        # att_map = torch.matmul(att_map, self.att_weight12)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _project_master(self, x, master, att_map):

        x1 = self.proj_with_attM(torch.matmul(
            att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.landmark_config = config["landmarks"]

        # Global CNN for each landmark
        self.globalCNN = nn.Sequential(
            ResidualBlock(3, 16),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.hidden_dim = 32
        # GRU for each landmark group
        self.gru_dict = nn.ModuleDict({
            key: nn.GRU(16, self.hidden_dim, batch_first=True)
            for key in self.landmark_config.keys()
        })

        self.landmark_dim = config['LandmarkGAT']['dim']
        self.dropout = config['LandmarkGAT']['dropout']
        # GAT for each landmark group
        self.gat_dict = nn.ModuleDict({
            key: LandmarkGAT(in_dim=self.hidden_dim, hidden_dim1=self.landmark_dim[0], hidden_dim2=self.landmark_dim[1], out_dim=self.landmark_dim[2], dropout=self.dropout)
            for key in self.landmark_config.keys()
        })

        self.global_nodes = config["global_nodes"]
        self.global_node_pool = GraphPool(self.landmark_dim[-1], pooling_ratio=self.global_nodes*self.landmark_dim[-1]/2048) # 2048 is the number of global features

        # HtrgGAT
        self.HtrgGAT_dim = config['HtrgGAT']['dim']
        self.temp = config['HtrgGAT']['temperature']
        self.HtrgGAT11 = HtrgGraphAttentionLayer(self.landmark_dim[-1], self.HtrgGAT_dim, temperature=self.temp)
        self.HtrgGAT12 = HtrgGraphAttentionLayer(self.landmark_dim[-1], self.HtrgGAT_dim, temperature=self.temp)
        self.HtrgGAT21 = HtrgGraphAttentionLayer(self.HtrgGAT_dim, self.HtrgGAT_dim, temperature=self.temp)
        self.HtrgGAT22 = HtrgGraphAttentionLayer(self.HtrgGAT_dim, self.HtrgGAT_dim, temperature=self.temp)
        self.master1 = nn.Parameter(torch.randn(1, 1, self.HtrgGAT_dim))
        self.master2 = nn.Parameter(torch.randn(1, 1, self.HtrgGAT_dim))

        self.pool = GraphPool(self.HtrgGAT_dim, pooling_ratio=0.5)
        self.drop_final = nn.Dropout(0.2)

        self.output_layer = nn.Linear(5*self.HtrgGAT_dim, 1)

    def forward(self, patch_video, landmark, global_video):
        # patch feature extraction using global CNN
        batch_size, num_landmarks, num_frames, channels, patch_size, _ = patch_video.size()
        patch_video = patch_video.view(-1, channels, patch_size, patch_size)
        cnn_features = self.globalCNN(patch_video)
        cnn_features = cnn_features.view(batch_size, num_landmarks, num_frames, -1)  # (batch, landmarks, frames, 16)

        device = next(self.parameters()).device

        # feature extraction using GRU and landmark GAT construction 
        graph_nodes = {}
        for landmark_type, (start, end) in self.landmark_config.items():
            node_features = []
            positional_encoding = []

            for landmark_idx in range(start, end + 1):
                landmark_seq = cnn_features[:, landmark_idx, :, :]
                # logging.debug("landmark_seq shape: %s", landmark_seq.shape)
                gru_output, _ = self.gru_dict[landmark_type](landmark_seq)
                node_features.append(gru_output[:, -1, :])

                # landmark positional encoding
                landmark_x_y = landmark[:, :, landmark_idx, :] # (batch, frames, 2)
                # logging.debug("landmark_x_y shape: %s", landmark_x_y.shape)

                landmark_pos_vector = landmark_x_y.reshape(batch_size, -1) # (batch, frames*2)
                # logging.debug("landmark_pos_vector shape: %s", landmark_pos_vector.shape)
                # logging.debug("landmark_pos_vector: %s", landmark_pos_vector)
                positional_encoding.append(get_positional_encoding(landmark_pos_vector, device))

            # 노드 특징을 tensor로 병합하여 GAT의 입력으로 사용
            node_features = torch.stack(node_features, dim=1)  # (batch, num_landmarks_subset, hidden_dim)
            positional_encoding = torch.stack(positional_encoding, dim=1)  # (batch, num_landmarks_subset, hidden_dim)
            logging.debug("node_features shape: %s", node_features.shape)
            logging.debug("positional_encoding shape: %s", positional_encoding.shape)
            # node_features와 positional_encoding을 결합
            node_features = node_features + positional_encoding

            # 완전 연결된 edge_index 생성 (landmark 그룹 내 노드가 모두 연결됨)
            num_nodes = end - start + 1
            fully_connected_matrix = torch.ones(num_nodes, num_nodes, device=device)
            edge_index = dense_to_sparse(fully_connected_matrix)[0].long().to(device)  
            assert num_nodes == node_features.shape[1]

            # 각 landmark 그룹에 대해 GAT 적용
            gat_output = self.gat_dict[landmark_type](node_features, edge_index)
            logging.debug("gat_output shape: %s", gat_output.shape)
            # GAT 출력을 원래 형태로 복구
            graph_nodes[landmark_type] = gat_output
            logging.debug("graph_nodes[%s] shape: %s", landmark_type, graph_nodes[landmark_type].shape)

        # 최종 output을 tensor로 통합
        logging.debug("graph_nodes.keys(): %s", graph_nodes.keys())
        for key in graph_nodes.keys():
            logging.debug("graph_nodes[%s] shape: %s", key, graph_nodes[key].shape)
        landmark_output = torch.cat([graph_nodes[key] for key in graph_nodes.keys()], dim=1)
        
        global_video = global_video.view(batch_size, -1, landmark_output.size(-1))
        logging.debug("global_video node shape: %s", global_video.shape)
        global_nodes = self.global_node_pool(global_video)
        logging.debug("global_nodes shape: %s", global_nodes.shape)

        master1 = self.master1.expand(batch_size, -1, -1)
        master2 = self.master2.expand(batch_size, -1, -1)
        logging.debug("master1 shape: %s", master1.shape)
        logging.debug("master2 shape: %s", master2.shape)

        # Htrg GAT with Landmark Nodes and Global Nodes
        # Inference 1
        out_L1, out_G1, master1 = self.HtrgGAT11(landmark_output, global_nodes, master=master1)
        logging.debug("out_L1 shape: %s out_G1 shape: %s master1 shape: %s", out_L1.shape, out_G1.shape, master1.shape)
        out_L1 = self.pool(out_L1)
        out_G1 = self.pool(out_G1)
        master1 = self.pool(master1)

        out_L1_aug, out_G1_aug, master_aug = self.HtrgGAT12(out_L1, out_G1, master=master1)
        logging.debug("out_L1_aug shape: %s out_G1_aug shape: %s master_aug shape: %s", out_L1_aug.shape, out_G1_aug.shape, master_aug.shape)
        out_L1 = out_L1 + out_L1_aug
        out_G1 = out_G1 + out_G1_aug
        master1 = master1 + master_aug

        # Inference 2
        out_L2, out_G2, master2 = self.HtrgGAT21(landmark_output, global_nodes, master=master2)
        logging.debug("out_L2 shape: %s out_G2 shape: %s master2 shape: %s", out_L2.shape, out_G2.shape, master2.shape)
        out_L2 = self.pool(out_L2)
        out_G2 = self.pool(out_G2)
        master2 = self.pool(master2)

        out_L2_aug, out_G2_aug, master2_aug = self.HtrgGAT22(out_L2, out_G2, master=master2)
        logging.debug("out_L2_aug shape: %s out_G2_aug shape: %s master2_aug shape: %s", out_L2_aug.shape, out_G2_aug.shape, master2_aug.shape)
        out_L2 = out_L2 + out_L2_aug
        out_G2 = out_G2 + out_G2_aug
        master2 = master2 + master2_aug

        out_L1 = self.drop_final(out_L1)
        out_L2 = self.drop_final(out_L2)
        out_G1 = self.drop_final(out_G1)
        out_G2 = self.drop_final(out_G2)
        master1 = self.drop_final(master1)
        master2 = self.drop_final(master2)

        out_L = torch.max(out_L1, out_L2)
        out_G = torch.max(out_G1, out_G2)
        master = torch.max(master1, master2)

        L_max, _ = torch.max(torch.abs(out_L), dim=1)
        G_max, _ = torch.max(torch.abs(out_G), dim=1)
        L_avg = torch.mean(out_L, dim=1)
        G_avg = torch.mean(out_G, dim=1)

        last_hidden = torch.cat([L_max, G_max, L_avg, G_avg, master.squeeze(1)], dim=1)
        logging.debug("last_hidden shape: %s", last_hidden.shape)

        last_hidden = self.drop_final(last_hidden)
        output = self.output_layer(last_hidden)

        return output, last_hidden

logging.basicConfig(level=logging.INFO) # DEBUG/INFO

# config = {
#     "patch_size": 9,
#     "landmark_config":{
#         "jawline": (0, 16),
#         "left_eyebrow": (17, 21),
#         "right_eyebrow": (22, 26),
#         "nose": (27, 35),
#         "left_eye": (36, 41),
#         "right_eye": (42, 47),
#         "outer_lip": (48, 59),
#         "inner_lip": (60, 67)
#     },
#     "LandmarkGAT":{
#         "dim": [32, 16, 16],
#         "dropout": 0.2
#     },
#     "global_nodes": 32,
#     "HtrgGAT":{
#         "dim": 16,
#         "temperature": 100.0
#     }
# }

# with open("./config/config.yaml", "r") as file:
#     config = yaml.safe_load(file)
# model_config = config["model_config"]
# print(model_config)

# # cuda 설정
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# video = torch.rand(2, 16, 224, 224, 3).to(device)  # (batch, frames, height, width, channels)
# landmark = torch.rand(2, 16, 68, 2).to(device)  # (batch, frames, landmarks, x-y)
# patch_video = torch.rand(2, 68, 16, 3, 9, 9).to(device)  # (batch, landmarks, frames, channels, patch_size, patch_size)
# global_video = torch.rand(2, 2048).to(device)  # (batch, features)

# model = Model(model_config).to(device)
# output = model(patch_video, landmark, global_video)
# print(output)