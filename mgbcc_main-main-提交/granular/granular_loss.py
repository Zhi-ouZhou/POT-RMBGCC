import torch
import numpy as np
from granular.base import GranularBall, GBList, MVGBList
from granular.tools import relation_of_views_gblists, merge_tensors, relation_of_views_gblists_tensor


class GranularContrastiveLoss(torch.nn.Module):
    # 对比学习：让近邻球靠近，非近邻球远离
    # 近邻矩阵相当于标识出正样本和负样本
    def __init__(self, temperature=1.):
        super(GranularContrastiveLoss, self).__init__()
        self.t = temperature

    def forward(self, gblist):
        pos_mask = gblist.affinity()
        neg_mask = 1 - pos_mask
        num_ins = len(gblist)
        idx = torch.arange(0, num_ins)
        # 修正正样本对掩码
        pos_mask[idx, idx] = 0
        x = gblist.get_centers()
        # 计算相似度，这里就是矩阵相乘
        norm_x = torch.norm(x, p=2, dim=1, keepdim=True)
        sim_x = x @ x.T / (norm_x @ norm_x.T + 1e-12)
        # 考虑用cross entropy 重写
        sim_pos = pos_mask * sim_x / self.t
        sim_neg = neg_mask * sim_x / self.t
        exp_sim_neg = torch.sum(torch.exp(sim_neg), dim=1, keepdim=True).expand((num_ins, num_ins))
        expsum_sim = torch.exp(sim_pos) + exp_sim_neg
        # expsum_sim = exp_sim_neg
        loss = -(sim_pos - torch.log(expsum_sim) * pos_mask)

        avg_sim_pos = torch.sum(sim_pos) / torch.sum(pos_mask)
        avg_sim_neg = torch.sum(sim_neg) / (torch.sum(neg_mask))
        return torch.sum(torch.as_tensor(loss)) / num_ins, avg_sim_pos, avg_sim_neg


class MultiviewGCLoss(torch.nn.Module):
    def __init__(self, temperature=1.):
        super(MultiviewGCLoss, self).__init__()
        self.t = temperature

    def forward(self, views: MVGBList):
        device = views[0].data.device
        loss = torch.tensor(0., device=device)
        num_views = len(views)

        for i in range(num_views):
            mask_i_intra = torch.eye(len(views[i]), device=device)
            for j in range(i + 1, num_views):
                mask_j_intra = torch.eye(len(views[j]), device=device)

                # 1. 获取基础交集 Mask，并强制挂载到相同的设备(GPU)上
                mask_inter = relation_of_views_gblists_tensor(views[i], views[j]).to(device)

                # 2. 基于引领树层级 (Layer) 的软加权机制

                if hasattr(views[i], 'tree_layers') and hasattr(views[j], 'tree_layers'):
                    L_i = views[i].tree_layers
                    L_j = views[j].tree_layers

                    diff_matrix = torch.abs(L_i.unsqueeze(1) - L_j.unsqueeze(0))
                    # 层级差异越大，拉近权重越小 (指数衰减)
                    weight_matrix = torch.exp(-0.5 * diff_matrix)

                    # 根节点激励：都是根节点(Layer=0)，额外增加 0.5 拉力
                    root_boost = ((L_i == 0).float().unsqueeze(1) * (L_j == 0).float().unsqueeze(0)) * 0.5
                    weight_matrix = weight_matrix + root_boost

                    # 将智能权重赋予交集 Mask
                    mask_inter = mask_inter * weight_matrix

                ni, nj = len(views[i]), len(views[j])
                # 合并掩码矩阵 (此时 pos_mask 内部含有 1.5, 0.6 等软权重值)
                pos_mask = merge_tensors(ni, nj, mask_i_intra, mask_inter, mask_inter.T, mask_j_intra).to(device)

                # 3. 提取纯粹的 0/1 掩码，用于内部安全的计算
                binary_pos_mask = (pos_mask > 0).float()
                neg_mask = 1.0 - binary_pos_mask

                num_ins = ni + nj
                centers_i = views[i].get_centers()
                centers_j = views[j].get_centers()
                x = torch.concat((centers_i, centers_j), dim=0)

                # 4. 计算余弦相似度
                norm_x = torch.norm(x, p=2, dim=1, keepdim=True)
                sim_x = x @ x.T / (norm_x @ norm_x.T + 1e-12)

                # 除以温度系数 t
                sim_scaled = sim_x / self.t

                # 5. 正样本相似度 (利用 0/1 掩码屏蔽无关项)
                sim_pos = binary_pos_mask * sim_scaled

                # 6. 先算 exp()，然后再用 neg_mask 屏蔽
                exp_sim = torch.exp(sim_scaled)
                # 计算负样本的指数和
                exp_sim_neg = torch.sum(exp_sim * neg_mask, dim=1, keepdim=True).expand((num_ins, num_ins))
                expsum_sim = torch.exp(sim_pos) + exp_sim_neg

                # 7. 计算出基础的 InfoNCE Loss 矩阵 (N x N)
                base_loss_matrix = -(sim_pos - torch.log(expsum_sim + 1e-12) * binary_pos_mask)

                # 8. 把带有 1.5、0.6 的软权重矩阵 (pos_mask) 乘在 Loss 外面
                loss += torch.sum(base_loss_matrix * pos_mask) / (pos_mask.sum() + 1e-12)

        return loss / (num_views * (num_views - 1) / 2)


class HierarchicalTreeLoss(torch.nn.Module):
    """
    引领树偏序对比损失：迫使子粒球在隐空间中靠近其父粒球
    """

    def __init__(self, temperature=1.0):
        super(HierarchicalTreeLoss, self).__init__()
        self.t = temperature

    def forward(self, gblist, tree_mask_np):

        device = gblist.data.device
        # 这里的 centers 是带有梯度的
        x = gblist.get_centers()
        K = len(x)

        pos_mask = torch.from_numpy(tree_mask_np).to(device)
        # 如果某个节点是根节点(没有父节点)，或者当前没有边，避免除零错误
        has_parent = pos_mask.sum(dim=1) > 0
        if not has_parent.any():
            return torch.tensor(0., device=device, requires_grad=True)

        # 计算余弦相似度
        norm_x = torch.norm(x, p=2, dim=1, keepdim=True)
        sim_x = x @ x.T / (norm_x @ norm_x.T + 1e-12)

        # 将自身从负样本中排除
        neg_mask = torch.ones((K, K), device=device) - torch.eye(K, device=device) - pos_mask

        sim_pos = pos_mask * sim_x / self.t
        sim_neg = neg_mask * sim_x / self.t

        # 只计算有父节点的行的 Loss
        exp_sim_neg = torch.sum(torch.exp(sim_neg), dim=1, keepdim=True)
        expsum_sim = torch.exp(sim_pos) + exp_sim_neg

        # InfoNCE
        loss_matrix = -(sim_pos - torch.log(expsum_sim + 1e-12) * pos_mask)

        # 只平均那些确实有父节点的子节点的损失
        valid_loss = torch.sum(loss_matrix) / (torch.sum(pos_mask) + 1e-12)
        return valid_loss



