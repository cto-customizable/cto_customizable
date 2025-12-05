import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BaseRecModel import BaseRecModel


class NGCF(BaseRecModel):
    @staticmethod
    def parse_model_args(parser, model_name='NGCF'):
        parser = BaseRecModel.parse_model_args(parser, model_name)
        parser.add_argument('--layer_sizes', type=str, default='64,64',
                            help='NGCF layer sizes')
        parser.add_argument('--mess_dropout', type=float, default=0.1,
                            help='message dropout')
        return parser

    def __init__(self, data_processor_dict, user_num, item_num,
                 u_vector_size, i_vector_size,
                 layer_sizes='64,64',
                 mess_dropout=0.1,
                 random_seed=2020,
                 dropout=0,
                 model_path='../model/NGCF/NGCF.pt'):
        """
        Args:
            data_processor_dict 需要包含 adjacency_matrix: shape [N, N] 的稀疏图
        """
        super(NGCF, self).__init__(
            data_processor_dict,
            user_num, item_num,
            u_vector_size, i_vector_size,
            random_seed, dropout, model_path
        )

        self.n_users = user_num
        self.n_items = item_num
        self.n_nodes = user_num + item_num
        self.adj_matrix = data_processor_dict['adjacency_matrix']   # 稀疏矩阵 (torch.sparse)

        self.layer_sizes = [int(x) for x in layer_sizes.split(',')]
        self.mess_dropout = mess_dropout

        self._init_nn()

    # -----------------------------------------------------
    # 初始化参数
    # -----------------------------------------------------
    def _init_nn(self):
        # 节点嵌入：user 和 item 拼一起 [n_nodes, embed_dim]
        self.embedding_user = nn.Embedding(self.n_users, self.u_vector_size)
        self.embedding_item = nn.Embedding(self.n_items, self.i_vector_size)

        embed_dim = self.u_vector_size
        self.layers = nn.ModuleList()
        last_dim = embed_dim

        for layer_dim in self.layer_sizes:
            W1 = nn.Parameter(torch.randn(last_dim, layer_dim) * 0.01)
            W2 = nn.Parameter(torch.randn(last_dim, layer_dim) * 0.01)
            self.layers.append(nn.ParameterList([W1, W2]))
            last_dim = layer_dim

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(self.mess_dropout)

    # -----------------------------------------------------
    # NGCF 消息传播
    # -----------------------------------------------------
    def propagate(self):
        # 初始节点嵌入 [N, d]
        user_embed = self.embedding_user.weight
        item_embed = self.embedding_item.weight
        ego_embeddings = torch.cat([user_embed, item_embed], dim=0)

        all_embeddings = [ego_embeddings]

        A = self.adj_matrix

        for W1, W2 in self.layers:
            # (1) t = A * E
            side_embeddings = torch.sparse.mm(A, ego_embeddings)

            # (2) W1 * (A E)
            sum_embeddings = torch.matmul(side_embeddings, W1)

            # (3) W2 * (E ⊙ A E)
            interaction_embeddings = ego_embeddings * side_embeddings
            bi_embeddings = torch.matmul(interaction_embeddings, W2)

            ego_embeddings = sum_embeddings + bi_embeddings
            ego_embeddings = self.leaky_relu(ego_embeddings)
            ego_embeddings = self.dropout_layer(ego_embeddings)

            all_embeddings.append(ego_embeddings)

        # 拼接所有层
        all_embeddings = torch.cat(all_embeddings, dim=1)

        # 划分回 user 和 item
        u_emb = all_embeddings[:self.n_users]
        i_emb = all_embeddings[self.n_users:]

        return u_emb, i_emb

    # -----------------------------------------------------
    # 预测
    # feed_dict: {uid, pos, neg}
    # -----------------------------------------------------
    def predict(self, feed_dict):
        check_list = []

        u_emb, i_emb = self.propagate()

        users = feed_dict['uid']
        pos_items = feed_dict['pos']
        neg_items = feed_dict['neg']

        u_vec = u_emb[users]                    # [B, d]
        pos_vec = i_emb[pos_items]              # [B, pos_k, d]
        neg_vec = i_emb[neg_items]              # [B, neg_k, d]

        pos_pred = (u_vec.unsqueeze(1) * pos_vec).sum(-1)
        neg_pred = (u_vec.unsqueeze(1) * neg_vec).sum(-1)

        prediction = torch.cat([pos_pred, neg_pred], dim=-1)

        return {
            'pos_prediction': pos_pred,
            'neg_prediction': neg_pred,
            'prediction': prediction,
            'check': check_list,
            'u_vectors': u_vec
        }

    # -----------------------------------------------------
    # predict_vectors: 用户向量外部提供（例如个性化优化）
    # -----------------------------------------------------
    def predict_vectors(self, vectors, feed_dict):
        check_list = []

        _, i_emb = self.propagate()

        pos_items = feed_dict['pos']
        neg_items = feed_dict['neg']

        u_vec = vectors
        pos_vec = i_emb[pos_items]
        neg_vec = i_emb[neg_items]

        pos_pred = (u_vec.unsqueeze(1) * pos_vec).sum(-1)
        neg_pred = (u_vec.unsqueeze(1) * neg_vec).sum(-1)
        prediction = torch.cat([pos_pred, neg_pred], -1)

        return {
            'pos_prediction': pos_pred,
            'neg_prediction': neg_pred,
            'prediction': prediction,
            'check': check_list,
            'u_vectors': u_vec
        }
