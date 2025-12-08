import torch
import torch.nn as nn
from models.BaseRecModel import BaseRecModel


class BPR(BaseRecModel):
    @staticmethod
    def parse_model_args(parser, model_name='BPR'):
        parser = BaseRecModel.parse_model_args(parser, model_name)
        return parser

    def __init__(self,
                 data_processor_dict,
                 user_num,
                 item_num,
                 u_vector_size,
                 i_vector_size,
                 random_seed=2020,
                 dropout=0,
                 model_path='../model/BPR/BPR.pt'):
        super(BPR, self).__init__(data_processor_dict=data_processor_dict,
                                  user_num=user_num,
                                  item_num=item_num,
                                  u_vector_size=u_vector_size,
                                  i_vector_size=i_vector_size,
                                  random_seed=random_seed,
                                  dropout=dropout,
                                  model_path=model_path)

    # -----------------------------------------------------
    # 初始化：user embedding + item embedding
    # -----------------------------------------------------
    def _init_nn(self):
        self.uid_embeddings = nn.Embedding(self.user_num, self.u_vector_size)
        self.iid_embeddings = nn.Embedding(self.item_num, self.i_vector_size)

    # -----------------------------------------------------
    # 标准 BPR：用户向量 · 物品向量
    # feed_dict 需要包含：
    #   uid: [B]
    #   pos: [B, n_pos]
    #   neg: [B, n_neg]
    # -----------------------------------------------------
    def predict(self, feed_dict):
        check_list = []

        u_ids = feed_dict['uid']          # [B]
        pos_ids = feed_dict['pos']        # [B, n_pos]
        neg_ids = feed_dict['neg']        # [B, n_neg]

        u_vec = self.uid_embeddings(u_ids)        # [B, d]
        pos_vec = self.iid_embeddings(pos_ids)    # [B, n_pos, d]
        neg_vec = self.iid_embeddings(neg_ids)    # [B, n_neg, d]

        pos_pred = (u_vec.unsqueeze(1) * pos_vec).sum(dim=-1)   # [B, n_pos]
        neg_pred = (u_vec.unsqueeze(1) * neg_vec).sum(dim=-1)   # [B, n_neg]

        pred = torch.cat([pos_pred, neg_pred], dim=-1)

        return {
            'pos_prediction': pos_pred,
            'neg_prediction': neg_pred,
            'prediction': pred,
            'check': check_list,
            'u_vectors': u_vec,
        }

    # -----------------------------------------------------
    # 外部提供用户向量 —— 与 BaseRecModel 兼容
    # -----------------------------------------------------
    def predict_vectors(self, vectors, feed_dict):
        check_list = []

        pos_ids = feed_dict['pos']
        neg_ids = feed_dict['neg']

        u_vec = vectors                          # [B, d]
        pos_vec = self.iid_embeddings(pos_ids)   # [B, n_pos, d]
        neg_vec = self.iid_embeddings(neg_ids)   # [B, n_neg, d]

        pos_pred = (u_vec.unsqueeze(1) * pos_vec).sum(dim=-1)
        neg_pred = (u_vec.unsqueeze(1) * neg_vec).sum(dim=-1)
        pred = torch.cat([pos_pred, neg_pred], dim=-1)

        return {
            'pos_prediction': pos_pred,
            'neg_prediction': neg_pred,
            'prediction': pred,
            'check': check_list,
            'u_vectors': u_vec,
        }
