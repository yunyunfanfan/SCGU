import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from . import RGCN


class DeletionLayer(nn.Module):
    def __init__(self, dim, mask):
        super().__init__()
        self.dim = dim
        self.mask = mask
        self.deletion_weight = nn.Parameter(torch.ones(dim, dim) / 1000)
        # self.deletion_weight = nn.Parameter(torch.eye(dim, dim))
        # init.xavier_uniform_(self.deletion_weight)
    
    def forward(self, x, mask=None):
        '''Only apply deletion operator to the local nodes identified by mask'''

        if mask is None:
            mask = self.mask
        
        if mask is not None:
            new_rep = x.clone()
            new_rep[mask] = torch.matmul(new_rep[mask], self.deletion_weight)

            return new_rep

        return x

class DeletionLayerKG(nn.Module):
    def __init__(self, dim, mask):
        super().__init__()
        self.dim = dim
        self.mask = mask
        self.deletion_weight = nn.Parameter(torch.ones(dim, dim) / 1000)
    
    def forward(self, x, mask=None):
        '''Only apply deletion operator to the local nodes identified by mask'''

        if mask is None:
            mask = self.mask
        
        if mask is not None:
            new_rep = x.clone()
            new_rep[mask] = torch.matmul(new_rep[mask], self.deletion_weight)

            return new_rep

        return x

class LowRankDeletionLayerKG(nn.Module):
    def __init__(self, dim, mask, num_relations, rank=16, init_strategy='random'):
        super().__init__()
        self.dim = dim
        self.mask = mask
        self.num_relations = num_relations
        self.rank = rank
        

        self.A = nn.Parameter(torch.randn(dim, rank) * 0.1)
        

        if init_strategy == 'zero':
            self.B = nn.Parameter(torch.zeros(num_relations, rank, dim))
            print("Using zero initialization for relation-specific B matrices")
        else:
            self.B = nn.Parameter(torch.randn(num_relations, rank, dim) * 0.1)
            print("Using random initialization for relation-specific B matrices")
        
        print(f"LowRankDeletionLayerKG: dim={dim}, rank={rank}, num_relations={num_relations}")
        print(f"Parameters: A={self.A.shape}, B={self.B.shape}")
        print("Using shared A matrix and relation-specific B matrices")
   
        total_params = self.A.numel() + self.B.numel()
        shared_params = dim * rank 
        relation_params = num_relations * rank * dim 
        print(f"Total parameters: {total_params:,}")
        print(f"  A (shared): {shared_params:,}")
        print(f"  B (relation-specific): {relation_params:,}")
    
    def forward(self, x, mask=None, edge_type=None):
        '''Apply relation-aware low-rank deletion operator
        If edge_type is provided (1D tensor of edge relation types for current message-passing edges),
        compute a relation-weighted B (by frequency of relations present, ignoring reverse types),
        otherwise fall back to global average over all relations.
        '''
        if mask is None:
            mask = self.mask
        
        if mask is not None:
            new_rep = x.clone()

            identity = torch.eye(self.dim, device=x.device, dtype=x.dtype)
            
            if edge_type is not None:
     
                dir_mask = (edge_type >= 0) & (edge_type < self.num_relations)
                rel_ids = edge_type[dir_mask]
                if rel_ids.numel() == 0:

                    B_avg = self.B.mean(dim=0)
                else:
                    counts = torch.bincount(rel_ids, minlength=self.num_relations).float()
                    weights = counts / (counts.sum() + 1e-8)  # [num_relations]
                    # 频率加权平均 B
                    B_avg = (weights.view(-1, 1, 1) * self.B).sum(dim=0)  # [rank, dim]
                low_rank_update = torch.matmul(self.A, B_avg)
            else:

                B_avg = self.B.mean(dim=0)
                low_rank_update = torch.matmul(self.A, B_avg)
                
                
            deletion_weight = identity + low_rank_update  # [dim, dim]
            new_rep[mask] = torch.matmul(new_rep[mask], deletion_weight)
            
            return new_rep

        return x
    
    def get_deletion_weight(self, relation_id=None):

        identity = torch.eye(self.dim, device=self.A.device, dtype=self.A.dtype)
        
        if relation_id is not None and relation_id < self.num_relations:

            low_rank_update = torch.matmul(self.A, self.B[relation_id])
            return identity + low_rank_update
        else:

            B_avg = self.B.mean(dim=0)
            low_rank_update = torch.matmul(self.A, B_avg)
            return identity + low_rank_update
    
    def get_all_deletion_weights(self):

        identity = torch.eye(self.dim, device=self.A.device, dtype=self.A.dtype)
        weights = []
        for i in range(self.num_relations):
            low_rank_update = torch.matmul(self.A, self.B[i])
            weights.append(identity + low_rank_update)
        return torch.stack(weights)  # [num_relations, dim, dim]
    
    def get_relation_diversity(self):

        B_flat = self.B.view(self.num_relations, -1)  # [num_relations, rank*dim]

        similarities = torch.cosine_similarity(B_flat.unsqueeze(1), B_flat.unsqueeze(0), dim=2)
     
        mask = ~torch.eye(self.num_relations, dtype=torch.bool, device=similarities.device)
        avg_similarity = similarities[mask].mean()
        return avg_similarity.item()

class RGCNDelete(RGCN):
    def __init__(self, args, num_nodes, num_edge_type, mask_1hop=None, mask_2hop=None, **kwargs):
        super().__init__(args, num_nodes, num_edge_type)
#修改2
        use_lowrank = getattr(args, 'use_lowrank_deletion', False)
        lowrank_rank = getattr(args, 'lowrank_rank', 16)
        self._use_lowrank_deletion = use_lowrank
        
        if use_lowrank:
            print(f"Using LowRankDeletionLayerKG with rank={lowrank_rank}")
            self.deletion1 = LowRankDeletionLayerKG(args.hidden_dim, mask_1hop, num_edge_type, rank=lowrank_rank)
            self.deletion2 = LowRankDeletionLayerKG(args.out_dim, mask_2hop, num_edge_type, rank=lowrank_rank)
        else:
            self.deletion1 = DeletionLayer(args.hidden_dim, mask_1hop)
            self.deletion2 = DeletionLayer(args.out_dim, mask_2hop)
#修改2结尾
        self.node_emb.requires_grad = False
        self.conv1.requires_grad = False
        self.conv2.requires_grad = False

    def forward(self, x, edge_index, edge_type, mask_1hop=None, mask_2hop=None, return_all_emb=False):
        # 冻结所有RGCN层，但允许deletion层训练
        with torch.no_grad():
            x = self.node_emb(x)
            x1 = self.conv1(x, edge_index, edge_type)
            x = F.relu(x1)
            x2 = self.conv2(x, edge_index, edge_type)
        
        # 重新启用梯度以便deletion层能够训练
        x1_detached = x1.detach().requires_grad_(True)
        x2_detached = x2.detach().requires_grad_(True)
        
        # deletion操作现在可以正常计算梯度
#修改3开头
        # 调用deletion算子（低秩版本按关系类型选择不同的B矩阵）
        if getattr(self, '_use_lowrank_deletion', False):
            x1_final = self.deletion1(x1_detached, mask_1hop, edge_type)
            x2_final = self.deletion2(x2_detached, mask_2hop, edge_type)
        else:
            x1_final = self.deletion1(x1_detached, mask_1hop)
            x2_final = self.deletion2(x2_detached, mask_2hop)

        if return_all_emb:
            return x1_final, x2_final
        
        return x2_final
    
    def get_original_embeddings(self, x, edge_index, edge_type, return_all_emb=False):
        return super().forward(x, edge_index, edge_type, return_all_emb)
