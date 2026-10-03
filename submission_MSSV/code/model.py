"""model.py — MLP cho bài toán 7 lớp (xem README mục 3 và GUIDE, "Quy định kiến trúc"):

    x (B, 54) -> Linear(54, h1) -> ReLU -> [Dropout] -> Linear(h1, h2) -> ReLU -> [Dropout]
              -> ... -> Linear(h_last, 7) -> logits (B, 7)

  - Lớp cuối ra logit thô, KHÔNG softmax trong model (softmax nằm trong hàm mất mát).
  - Dropout chỉ đặt sau ReLU của lớp ẩn. Mọi nn.Linear đều có bias. Không BatchNorm, không residual.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Số tham số bắt buộc ứng với từng kiến trúc (in_features=54, num_classes=7)
EXPECTED_PARAMS = {
    (256, 128): 47_879,        # M-base  (baseline)
    (512, 256): 161_287,       # M-wide
    (256, 128, 64): 55_687,    # M-deep
}


class MLP(nn.Module):
    """MLP theo quy định ở đầu file.

    Args:
        hidden:   tuple số nơ-ron các lớp ẩn, ví dụ (256, 128)
        dropout:  xác suất TẮT nơ-ron q (đúng nghĩa p của nn.Dropout); 0.0 = không dùng
        init:     "zeros" | "normal" | "xavier" | "he" | "default"
    """

    def __init__(self, hidden=(256, 128), dropout: float = 0.0, init: str = "he",
                 in_features: int = 54, num_classes: int = 7):
        super().__init__()
        layers, d = [], in_features
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, num_classes))      # lớp ra: logit thô
        self.net = nn.Sequential(*layers)
        init_weights(self, init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 54) float32  ->  logits: (B, 7) float32."""
        return self.net(x)


def init_weights(model: nn.Module, init: str) -> None:
    """Khởi tạo tham số của MỌI nn.Linear (bias = 0, trừ "default").

    "zeros"  : W = 0
    "normal" : W ~ N(0, 0.01^2)
    "xavier" : nn.init.xavier_normal_ (Var = 2/(n_in+n_out))
    "he"     : nn.init.kaiming_normal_(w, nonlinearity="relu")  (Var = 2/n_in; mode mặc định fan_in)
    "default": giữ khởi tạo mặc định của nn.Linear (kaiming_uniform với a=sqrt(5); KHÔNG phải He)
    """
    if init not in ("zeros", "normal", "xavier", "he", "default"):
        raise ValueError(f"init không hợp lệ: {init}")
    if init == "default":
        return
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if init == "zeros":
                nn.init.zeros_(m.weight)
            elif init == "normal":
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            elif init == "xavier":
                nn.init.xavier_normal_(m.weight)
            else:
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)


def count_params(model: nn.Module) -> int:
    """Tổng số tham số huấn luyện được."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def activation_stats(model: nn.Module, x: torch.Tensor) -> list[float]:
    """Độ lệch chuẩn của đầu ra SAU MỖI nn.Linear (tiền-kích-hoạt; phần tử cuối là logits), ở bước 0.

    Dùng cho thí nghiệm khởi tạo: với He, std này giữ ổn định qua các lớp ẩn; với zeros là 0.
    """
    model.eval()
    h, stats = x, []
    for layer in model.net:
        h = layer(h)
        if isinstance(layer, nn.Linear):
            stats.append(h.std().item())
    return stats
