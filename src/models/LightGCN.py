import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BaseRecModel import BaseRecModel


class LightGCN(BaseRecModel):
    @staticmethod
    def parse_model_args(parser, model_name='LightGCN'):
        parser = BaseRecModel.parse_model_args(parser, model_name)
        parser.add_argument('--layer_num', type=int, default=3,
                            help='Number of propagation layers')
        return parser

    def __init__(self, data_processor_dict, user_num, item_num,
                 u_vector_size, i_vector_size,
                 layer_num=3,
                 random_seed=2020,
                 dropout=0,
                 model_path='../model/LightGCN/LightGCN.pt'):
        """
        data_processor_dict 需要包含：
            adjacency_matrix: shape [n_users + n_items, n_users + n_items] 的归一化邻接矩阵 A_hat（sparse）
        """
        self.layer_num = layer_num
        self.adj_matrix = data_processor_dict['adjacency_matrix']  # torch.sparse.FloatTensor

        super(LightGCN, self).__init__(
            data_processor_dict=data_processor_dict,
            user_num=user_num,
            item_num=item_num,
            u_vector_size=u_vector_size,
            i_vector_size=i_vector_size,
            random_seed=random_seed,
            dropout=dropout,
            model_path=model_path,
        )

    # -----------------------------------------------------
    # 初始化：LightGCN 只有 embedding
    # -----------------------------------------------------
    def _init_nn(self):
        self.embedding_user = nn.Embedding(self.user_num, self.u_vector_size)
        self.embedding_item = nn.Embedding(self.item_num, self.i_vector_size)

    # -----------------------------------------------------
    # LightGCN propagation: 多层平均
    # -----------------------------------------------------
    def propagate(self):
        user_emb = self.embedding_user.weight      # [U, d]
        item_emb = self.embedding_item.weight      # [I, d]
        all_emb = torch.cat([user_emb, item_emb], dim=0)  # [N, d]

        embs = [all_emb]  # 第 0 层（初始）

        A_hat = self.adj_matrix   # 归一化邻接矩阵 (sparse)

        for _ in range(self.layer_num):
            all_emb = torch.sparse.mm(A_hat, all_emb)     # LightGCN propagation
            embs.append(all_emb)

        # 多层平均 pooling
        final_emb = torch.stack(embs, dim=1).mean(dim=1)

        # 划分回 user / item
        u_final = final_emb[:self.user_num]
        i_final = final_emb[self.user_num:]

        return u_final, i_final

    # -----------------------------------------------------
    # 预测正负样本
    # feed_dict = {uid, pos, neg}
    # -----------------------------------------------------
    def predict(self, feed_dict):
        check_list = []

        u_emb, i_emb = self.propagate()

        users = feed_dict['uid']      # [B]
        pos_items = feed_dict['pos']  # [B, n_pos]
        neg_items = feed_dict['neg']  # [B, n_neg]

        u_vec = u_emb[users]              # [B, d]
        pos_vec = i_emb[pos_items]        # [B, n_pos, d]
        neg_vec = i_emb[neg_items]        # [B, n_neg, d]

        pos_pred = (u_vec.unsqueeze(1) * pos_vec).sum(-1)
        neg_pred = (u_vec.unsqueeze(1) * neg_vec).sum(-1)
        pred = torch.cat([pos_pred, neg_pred], dim=-1)

        return {
            'pos_prediction': pos_pred,
            'neg_prediction': neg_pred,
            'prediction': pred,
            'check': check_list,
            'u_vectors': u_vec
        }

    # -----------------------------------------------------
    # 外部提供用户向量 (如个性化优化)
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

        pred = torch.cat([pos_pred, neg_pred], dim=-1)

        return {
            'pos_prediction': pos_pred,
            'neg_prediction': neg_pred,
            'prediction': pred,
            'check': check_list,
            'u_vectors': u_vec
        }
