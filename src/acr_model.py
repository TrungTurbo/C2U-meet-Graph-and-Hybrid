#!/usr/bin/env python3
"""
acr_model.py
Định nghĩa mạng ACR (Atomic Concept Representation) và bộ 3 Loss theo đặc tả C2U:
- Ma trận từ điển A in R^{N x d} (chuẩn hóa L2 từng hàng)
- Projection Head f_theta nhẹ
- Trọng số nguyên tử: w_x = softmax(sim(h_x, A) / tau) với tau = 0.1
- Vector tái tạo: z_x = normalize(A^T w_x)
- Bộ 3 Loss: L = L_tri (m_r = 0.3) + lambda_align * L_align (1.0) + lambda_ent * L_ent (0.02)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class ProjectionHead(nn.Module):
    """Mạng chiếu nhẹ bảo toàn chiều biểu diễn d -> d với residual connection."""
    def __init__(self, d: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = d * 2
        self.net = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d)
        )
        self.norm = nn.LayerNorm(d)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection + L2 normalization
        h = self.norm(x + self.net(x))
        return F.normalize(h, p=2, dim=-1)


class ACRModel(nn.Module):
    """
    Atomic Concept Representation (ACR) Model.
    - A: Learned Dictionary matrix of shape (N, d)
    - projection_head: Light MLP projecting embedding into concept space
    """
    def __init__(self, d: int = 384, N: int = 768, tau: float = 0.1):
        super().__init__()
        self.d = d
        self.N = N
        self.tau = tau
        
        # Projection Head
        self.projection_head = ProjectionHead(d)
        
        # Dictionary matrix A (N atoms, each of dimension d)
        # Khởi tạo chuẩn và normalize L2 trên từng hàng
        init_A = torch.randn(N, d)
        init_A = F.normalize(init_A, p=2, dim=-1)
        self.A = nn.Parameter(init_A)
        
    def get_normalized_dictionary(self) -> torch.Tensor:
        """Đảm bảo các hàng của A luôn có chuẩn L2 = 1."""
        return F.normalize(self.A, p=2, dim=-1)
        
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Chiếu vector x qua projection head và L2 normalize."""
        return self.projection_head(x)
        
    def compute_weights(self, h: torch.Tensor) -> torch.Tensor:
        """
        Tính trọng số nguyên tử w_x = softmax(sim(h_x, A) / tau).
        h: shape (..., d)
        Trả về w: shape (..., N)
        """
        A_norm = self.get_normalized_dictionary()  # (N, d)
        h_norm = F.normalize(h, p=2, dim=-1)      # (..., d)
        # Cosine similarity: h_norm @ A_norm.T
        sim = torch.matmul(h_norm, A_norm.t())     # (..., N)
        w = F.softmax(sim / self.tau, dim=-1)
        return w
        
    def reconstruct(self, w: torch.Tensor) -> torch.Tensor:
        """
        Vector tái tạo z_x = normalize(A^T w_x) = normalize(w_x @ A).
        w: shape (..., N)
        Trả về z: shape (..., d)
        """
        A_norm = self.get_normalized_dictionary()  # (N, d)
        z = torch.matmul(w, A_norm)                # (..., d)
        return F.normalize(z, p=2, dim=-1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass:
        x -> h_proj -> atomic weights w -> reconstructed z
        """
        h_proj = self.project(x)
        w = self.compute_weights(h_proj)
        z = self.reconstruct(w)
        return h_proj, w, z


class ACRLoss(nn.Module):
    """
    Bộ 3 hàm mất mát cho ACR:
    L = L_tri (m_r = 0.3) + lambda_align * L_align (1.0) + lambda_ent * L_ent (0.02)
    """
    def __init__(
        self,
        margin: float = 0.3,
        lambda_align: float = 1.0,
        lambda_ent: float = 0.02
    ):
        super().__init__()
        self.margin = margin
        self.lambda_align = lambda_align
        self.lambda_ent = lambda_ent
        
    def cosine_triplet_loss(
        self,
        q_proj: torch.Tensor,
        pos_proj: torch.Tensor,
        neg_proj: torch.Tensor
    ) -> torch.Tensor:
        """
        Cosine Triplet Loss: max(0, sim(q, neg) - sim(q, pos) + margin)
        q, pos, neg đều đã được L2 normalized.
        """
        sim_pos = torch.sum(q_proj * pos_proj, dim=-1)
        sim_neg = torch.sum(q_proj * neg_proj, dim=-1)
        loss = F.relu(sim_neg - sim_pos + self.margin)
        return loss.mean()
        
    def cosine_alignment_loss(
        self,
        w_q: torch.Tensor,
        w_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Cosine Alignment Loss giữa phân bố trọng số nguyên tử của query và pos_doc:
        1.0 - cos_sim(w_q, w_pos)
        """
        w_q_norm = F.normalize(w_q, p=2, dim=-1)
        w_pos_norm = F.normalize(w_pos, p=2, dim=-1)
        cos_sim = torch.sum(w_q_norm * w_pos_norm, dim=-1)
        return (1.0 - cos_sim).mean()
        
    def entropy_penalty(self, w: torch.Tensor) -> torch.Tensor:
        """
        Entropy penalty: H(w) = - sum(w * log(w + eps))
        """
        eps = 1e-9
        ent = -torch.sum(w * torch.log(w + eps), dim=-1)
        return ent.mean()
        
    def forward(
        self,
        q_proj: torch.Tensor,
        pos_proj: torch.Tensor,
        neg_proj: torch.Tensor,
        w_q: torch.Tensor,
        w_pos: torch.Tensor,
        w_neg: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        l_trip = self.cosine_triplet_loss(q_proj, pos_proj, neg_proj)
        l_align = self.cosine_alignment_loss(w_q, w_pos)
        l_ent = (self.entropy_penalty(w_q) + self.entropy_penalty(w_pos) + self.entropy_penalty(w_neg)) / 3.0
        
        total_loss = l_trip + (self.lambda_align * l_align) + (self.lambda_ent * l_ent)
        
        return {
            "total_loss": total_loss,
            "triplet_loss": l_trip,
            "align_loss": l_align,
            "ent_loss": l_ent
        }
