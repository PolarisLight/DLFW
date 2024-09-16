import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import core_module

def dis_2_score(dis, return_numpy=True):
    """
    convert distance to score
    :param dis: distance
    :return: score
    """
    w = torch.linspace(1, 10, 10).to(dis.device)
    w_batch = w.repeat(dis.shape[0], 1)
    score = (dis * w_batch).sum(dim=1)
    if return_numpy:
        return score.cpu().numpy()
    else:
        return score



class SupCon(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupCon, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """
        Calculate the supervised contrastive loss

        :param embeddings: Tensor of shape (batch_size, feature_dim), the feature embeddings.
        :param labels: Tensor of shape (batch_size,), the labels of the embeddings.
        :return: The supervised contrastive loss.
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        # Compute the cosine similarity matrix
        cosine_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1) / self.temperature

        # Mask for removing positive diagonal elements
        diag_mask = ~(torch.eye(batch_size, device=device).bool())

        # Exponential mask for the numerator
        exp_logits = torch.exp(cosine_sim) * diag_mask

        # Compute sum of exp logits
        log_prob = exp_logits / exp_logits.sum(dim=1, keepdim=True)

        # Create mask for positive samples
        labels_eq = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # Compute Supervised Contrastive Loss
        loss = -(labels_eq * diag_mask).float() * torch.log(log_prob + 1e-8)
        loss = loss.sum() / batch_size
        # loss = F.sigmoid(loss)

        return loss


class SupCRLoss(nn.Module):
    def __init__(self, margin=1.0, temperature=0.07, m=None):
        """
        Initialize the Supervised Contrastive Regression Loss module.
        :param margin: Margin to define the boundary for dissimilar targets.
        :param temperature: Temperature scaling to control the separation of embeddings.
        """
        super(SupCRLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
        self.m = m

    def forward(self, embeddings, targets):
        """
        Forward pass to compute the SupCR loss.
        :param embeddings: Tensor of shape (batch_size, embedding_dim), embedding representations of inputs.
        :param targets: Tensor of shape (batch_size,), continuous target values associated with each input.
        :return: The computed SupCR loss.
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute pairwise cosine similarity
        sim_matrix = torch.matmul(embeddings, embeddings.T)

        # Compute target differences matrix
        target_diffs = targets.unsqueeze(1) - targets.unsqueeze(0)

        # Apply margin to target differences
        target_diffs = torch.abs(target_diffs) - self.margin
        target_diffs = torch.clamp(target_diffs, min=0.0)

        # Calculate positive and negative masks
        positive_mask = target_diffs.eq(0).float() - torch.eye(target_diffs.shape[0], device=embeddings.device).float()
        negative_mask = target_diffs.gt(0).float()

        # Compute loss
        loss_positives = -torch.log(torch.exp(sim_matrix * positive_mask / self.temperature) + 1e-6).mean()
        loss_negatives = torch.log(torch.exp(sim_matrix * negative_mask / self.temperature) + 1e-6).mean()

        if self.m:
            loss = self.m - loss_positives + loss_negatives
        else:
            loss = loss_positives + loss_negatives

        return loss
    
class emd_loss(nn.Module):
    """
    Earth Mover Distance loss
    """

    def __init__(self, dist_r=2, use_l1loss=True, l1loss_coef=0.0):
        super(emd_loss, self).__init__()
        self.dist_r = dist_r
        self.use_l1loss = use_l1loss
        self.l1loss_coef = l1loss_coef

    def check_type_forward(self, in_types):
        assert len(in_types) == 2
        x_type, y_type = in_types
        assert x_type.size()[0] == y_type.shape[0]
        assert x_type.size()[0] > 0
        assert x_type.ndim == 2
        assert y_type.ndim == 2

    def forward(self, x, y_true):
        """
        Calculate EMD (Earth Mover's Distance) between predicted and true distributions.
        """
        self.check_type_forward((x, y_true))

        cdf_x = torch.cumsum(x, dim=-1)
        cdf_ytrue = torch.cumsum(y_true, dim=-1)
        if self.dist_r == 2:
            samplewise_emd = torch.sqrt(torch.mean(torch.pow(cdf_ytrue - cdf_x, 2), dim=-1))
        else:
            samplewise_emd = torch.mean(torch.abs(cdf_ytrue - cdf_x), dim=-1)
        
        # Calculate final EMD loss
        loss = torch.mean(samplewise_emd)

        if self.use_l1loss:
            rate_scale = torch.tensor([float(i + 1) for i in range(x.size()[1])], dtype=x.dtype, device=x.device)
            x_mean = torch.mean(x * rate_scale, dim=-1)
            y_true_mean = torch.mean(y_true * rate_scale, dim=-1)
            l1loss = torch.mean(torch.abs(x_mean - y_true_mean))
            loss += l1loss * self.l1loss_coef

        return loss, samplewise_emd  # Return EMD loss and EMD matrix for further use in SupCon

@core_module
class SupConEMD(nn.Module):
    """
    Supervised Contrastive Loss using EMD similarity as a basis for positive pair selection.
    """
    def __init__(self,args):
        super(SupConEMD, self).__init__()
        assert vars(args.core_params).keys() >= {'temperature', 'emd_threshold', 'weight_supcon'}, "Missing core parameters"
        # ================== Core Parameters ==================
        self.temperature = args.core_params.temperature
        self.emd_threshold = args.core_params.emd_threshold
        self.weight_supcon = args.core_params.weight_supcon
        # =====================================================
        self.emd_loss_fn = emd_loss() # EMD loss function

    def get_core_params(self)->dict[str, list[float]]:
        return {'weight_supcon': [0.0, 1.0],
                'temperature': [0.0, 1.0],
                'emd_threshold': [0.0, 1.0]}
        

    def forward(self, embeddings, pred_dist, true_dist):
        """
        Calculate the supervised contrastive loss for a distribution prediction task using EMD.

        :param embeddings: Tensor of shape (batch_size, feature_dim), the feature embeddings.
        :param pred_dist: Predicted distribution for each sample.
        :param true_dist: True distribution for each sample.
        :return: The supervised contrastive loss and EMD loss.
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        # Compute the cosine similarity matrix for embeddings
        cosine_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1) / self.temperature

        # Mask for removing positive diagonal elements (i.e., self-contrast)
        diag_mask = ~(torch.eye(batch_size, device=device).bool())

        # Exponential of similarity matrix for all pairs
        exp_logits = torch.exp(cosine_sim) * diag_mask

        # Sum of exponentiated logits (denominator for log-probability)
        log_prob = exp_logits / exp_logits.sum(dim=1, keepdim=True)

        # Calculate EMD between predicted and true distributions
        emd_loss_val, emd_matrix = self.emd_loss_fn(pred_dist, true_dist)

        # Create mask for positive pairs based on EMD (lower distance = more similar)
        pos_mask = (emd_matrix < self.emd_threshold).float() * diag_mask.float()

        # Compute supervised contrastive loss
        loss_supcon = -(pos_mask * torch.log(log_prob + 1e-8)).sum(dim=1)

        # Normalize loss by number of positive samples per anchor
        num_pos_samples = pos_mask.sum(dim=1)
        loss_supcon = loss_supcon / num_pos_samples.clamp(min=1)

        # Average loss over batch
        loss_supcon = loss_supcon.mean()

        loss = self.weight_supcon * emd_loss_val + (1 - self.weight_supcon) * loss_supcon

        return {"loss": loss, "emd_loss": emd_loss_val, "supcon_loss": loss_supcon}

class SupConEMDWithScore(nn.Module):
    """
    Supervised Contrastive Loss using EMD similarity and score-based classification.
    """

    def __init__(self, temperature=0.07, emd_threshold=0.1):
        super(SupConEMDWithScore, self).__init__()
        self.temperature = temperature
        self.emd_threshold = emd_threshold
        self.emd_loss_fn = emd_loss()

    def forward(self, embeddings, pred_dist, true_dist):
        device = embeddings.device
        batch_size = embeddings.shape[0]

        # 计算嵌入的余弦相似度
        cosine_sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1) / self.temperature
        diag_mask = ~(torch.eye(batch_size, device=device).bool())
        exp_logits = torch.exp(cosine_sim) * diag_mask
        log_prob = exp_logits / exp_logits.sum(dim=1, keepdim=True)

        # 计算 EMD 损失
        emd_loss_val, emd_matrix = self.emd_loss_fn(pred_dist, true_dist)

        # 选择基于 EMD 的正样本
        pos_mask = (emd_matrix < self.emd_threshold).float() * diag_mask.float()

        # 计算分数并转换为 1-10 的类别
        pred_score = dis_2_score(pred_dist, return_numpy=False)  # 得到预测分数
        true_score = dis_2_score(true_dist, return_numpy=False)  # 得到真实分数

        # 将分数量化到 1-10 的类别
        pred_labels = torch.clamp(pred_score.round(), min=1, max=10).long()
        true_labels = torch.clamp(true_score.round(), min=1, max=10).long()

        # 计算 Supervised Contrastive Loss，基于分数类别
        labels_eq = (pred_labels.unsqueeze(0) == true_labels.unsqueeze(1)).float()

        # 使用分数相同的样本作为正样本
        pos_mask_score = labels_eq * diag_mask.float()

        # 计算 SupCon 损失
        loss_supcon = -(pos_mask_score * torch.log(log_prob + 1e-8)).sum(dim=1)
        num_pos_samples = pos_mask_score.sum(dim=1)
        loss_supcon = loss_supcon / num_pos_samples.clamp(min=1)
        loss_supcon = loss_supcon.mean()

        return loss_supcon, emd_loss_val
if __name__ == "__main__":
    supcon = SupConEMD()
    feat = torch.randn(128, 768)
    pred = torch.rand(128, 10)
    labels = torch.rand(128, 10)
    loss = supcon(feat, pred, labels)
    print(loss)
