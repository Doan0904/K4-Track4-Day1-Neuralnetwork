"""data.py — nạp dữ liệu CoverType đã chia sẵn, tách validation, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu:
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối: không dùng để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ .npz. Trả về X_train_full, y_train_full, X_eval, y_eval, eval_row_id."""
    tr = np.load(f"{processed_dir}/train.npz")
    ev = np.load(f"{processed_dir}/eval.npz")
    X_tr, y_tr = tr["X"], tr["y"]
    X_ev, y_ev, row_id = ev["X"], ev["y"], ev["row_id"]
    for X, y in ((X_tr, y_tr), (X_ev, y_ev)):
        assert X.ndim == 2 and X.shape[1] == 54 and X.dtype == np.float32
        assert y.shape == (len(X),) and y.dtype == np.int64 and y.min() >= 0 and y.max() <= 6
    assert len(row_id) == len(X_ev)
    return X_tr, y_tr, X_ev, y_ev, row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (phân tầng theo nhãn). Cùng seed/val_fraction cho mọi thí nghiệm."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed)
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """mean/std của 10 cột số, CHỈ trên train (sau khi tách val).

    Vì sao không dùng val/eval: thống kê của chúng sẽ "rò" thông tin của dữ liệu chưa thấy vào
    bước tiền xử lý, làm điểm val/eval lạc quan và (với eval) vi phạm luật chơi.
    """
    mean = X_tr[:, :N_NUMERIC].mean(axis=0)
    std = X_tr[:, :N_NUMERIC].std(axis=0)
    return mean, std


def apply_standardizer(X, mean, std):
    """Bản sao của X với 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên."""
    out = X.copy()
    out[:, :N_NUMERIC] = (out[:, :N_NUMERIC] - mean) / np.where(std > 0, std, 1.0)
    return out


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed", verbose: bool = True) -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict: X_tr, y_tr, X_val, y_val, X_eval, y_eval (tensor trên device, y int64),
    eval_row_id (numpy), mean, std, majority_val_acc.
    """
    X_full, y_full, X_ev, y_ev, row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_full, y_full, val_fraction, seed)
    mean, std = fit_standardizer(X_tr)                       # chỉ trên train còn lại
    X_tr, X_val, X_ev = (apply_standardizer(X, mean, std) for X in (X_tr, X_val, X_ev))

    def f(a):
        return torch.tensor(a, dtype=torch.float32, device=device)

    def l(a):
        return torch.tensor(a, dtype=torch.int64, device=device)

    major = int(np.bincount(y_tr, minlength=7).argmax())
    maj_acc = float((y_val == major).mean())
    data = dict(X_tr=f(X_tr), y_tr=l(y_tr), X_val=f(X_val), y_val=l(y_val),
                X_eval=f(X_ev), y_eval=l(y_ev), eval_row_id=row_id,
                mean=mean, std=std, majority_val_acc=maj_acc)
    if verbose:
        print(f"X_tr {tuple(data['X_tr'].shape)} | X_val {tuple(data['X_val'].shape)} | "
              f"X_eval {tuple(data['X_eval'].shape)}")
        print(f"'luôn đoán lớp đa số' (lớp {major}) trên val: acc = {maj_acc:.4f}")
    return data


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Batch cuối có thể nhỏ hơn batch_size; ta GIỮ nó (không drop_last) để mọi mẫu được dùng mỗi epoch.
    """
    n = len(X)
    if shuffle:
        perm = torch.randperm(n, generator=generator, device=X.device)
    else:
        perm = torch.arange(n, device=X.device)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
