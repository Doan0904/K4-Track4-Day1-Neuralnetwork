"""train.py — đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.

Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).
Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import copy
import json
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params, activation_stats
from optimizer import build_optimizer, build_scheduler, clip_gradients

N_CLASSES = 7
TRAIN_EVAL_SUBSET = 50_000   # train loss/acc/F1 mỗi epoch đo (eval mode) trên tập con CỐ ĐỊNH này

# Cấu hình mặc định = BASELINE (M-base). `lr` được chọn bằng val ở notebook (Part 2).
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # chọn bằng val, không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    scheduler=None,            # None | "cosine" (cosine theo từng bước, T_max = tổng số bước)
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.
    cm: (7, 7), hàng = nhãn thật, cột = dự đoán (giống scripts/evaluate.py)."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Nhãn dự đoán int64 (N,) = argmax logits, ở chế độ eval() (dropout tắt)."""
    model.eval()
    return torch.cat([model(X[i:i + batch_size]).argmax(dim=1) for i in range(0, len(X), batch_size)])


def compute_loss(logits, y, loss_name: str, reduction: str = "mean"):
    """"ce" : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse": MSE giữa logit (KHÔNG softmax) và one-hot của y, trung bình trên MỌI phần tử (B*7),
              đúng nn.MSELoss / F.mse_loss mặc định (không có hệ số 1/2)."""
    if loss_name == "ce":
        return F.cross_entropy(logits, y, reduction=reduction)
    if loss_name == "mse":
        return F.mse_loss(logits, F.one_hot(y, N_CLASSES).to(logits.dtype), reduction=reduction)
    raise ValueError(f"loss phải là 'ce' hoặc 'mse', nhận {loss_name!r}")


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """dict(loss, acc, macro_f1) ở chế độ eval() và no_grad (luôn FP32).
    loss = tổng loss (reduction="sum") chia số phần tử (N với CE, N*7 với MSE) => cùng thang với mean."""
    model.eval()
    total, cm = 0.0, torch.zeros(N_CLASSES * N_CLASSES, dtype=torch.int64, device=X.device)
    for i in range(0, len(X), batch_size):
        xb, yb = X[i:i + batch_size], y[i:i + batch_size]
        logits = model(xb)
        total += compute_loss(logits, yb, loss_name, reduction="sum").item()
        cm += torch.bincount(yb * N_CLASSES + logits.argmax(1), minlength=N_CLASSES * N_CLASSES)
    cm = cm.view(N_CLASSES, N_CLASSES).cpu().numpy()
    n = len(X) * (N_CLASSES if loss_name == "mse" else 1)
    return dict(loss=total / n, acc=float(np.trace(cm) / cm.sum()), macro_f1=macro_f1_from_confusion(cm))


def _finite_stats(a: np.ndarray):
    a = a[np.isfinite(a)]
    return a if len(a) else np.array([np.nan])


def run_experiment(cfg: dict, data: dict, verbose: bool = False) -> dict:
    """Huấn luyện một cấu hình và trả về {"cfg", "history", "summary", "best_state"}.

    - Train loss/acc mỗi epoch đo ở eval() trên tập con cố định 50 000 mẫu của train (so sánh được với val).
    - grad_norm = chuẩn L2 toàn cục TRƯỚC khi clip, trung bình các bước hữu hạn trong epoch.
    - Chọn epoch tốt nhất theo val_loss (như dừng sớm); val_acc/val_macro_f1 trong summary lấy ở epoch đó.
    - Chỉ dùng val; KHÔNG đưa X_eval vào đây.
    - Thời gian/epoch chỉ tính phần huấn luyện (không gồm đánh giá), có synchronize.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    cfg["hidden"] = tuple(cfg["hidden"])
    assert cfg["lr"] is not None, "cần đặt lr"
    X_tr, y_tr, X_val, y_val = data["X_tr"], data["y_tr"], data["X_val"], data["y_val"]
    device = X_tr.device
    use_cuda = device.type == "cuda"

    set_seed(cfg["seed"])
    model = MLP(cfg["hidden"], cfg["dropout"], cfg["init"]).to(device)
    assert count_params(model) == EXPECTED_PARAMS[cfg["hidden"]], count_params(model)
    opt = build_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"],
                          weight_decay=cfg["weight_decay"], momentum=cfg["momentum"])
    steps_per_epoch = -(-len(X_tr) // cfg["batch"])
    sched = build_scheduler(opt, cfg["scheduler"], cfg["epochs"] * steps_per_epoch)

    prec = cfg["precision"]
    amp_dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[prec]
    scaler = torch.amp.GradScaler("cuda") if (prec == "fp16" and use_cuda) else None
    if prec == "fp16" and not use_cuda:
        raise RuntimeError("fp16 autocast + GradScaler cần GPU CUDA")

    gen = torch.Generator(device=device)
    gen.manual_seed(cfg["seed"])
    sub = torch.randperm(len(X_tr), generator=torch.Generator().manual_seed(0))[:TRAIN_EVAL_SUBSET].to(device)
    X_sub, y_sub = X_tr[sub], y_tr[sub]

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    step0 = evaluate(model, X_val, y_val, cfg["loss"])["loss"]          # TRƯỚC bước cập nhật đầu tiên
    act0 = activation_stats(model, X_val[:512])

    hist = {k: [] for k in ("epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1",
                            "grad_norm", "epoch_time_s")}
    best = dict(val_loss=float("inf"), epoch=None, acc=None, f1=None, state=None)
    all_gn, diverged = [], False
    t_run = time.time()

    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        loss_buf = torch.empty(steps_per_epoch, device=device)
        gn_buf = torch.empty(steps_per_epoch, device=device)
        for k, (xb, yb) in enumerate(iterate_batches(X_tr, y_tr, cfg["batch"], gen)):
            if amp_dtype is not None:
                with torch.autocast(device.type, dtype=amp_dtype):          # chỉ bọc forward + loss
                    logits = model(xb)
                    loss = compute_loss(logits.float(), yb, cfg["loss"])
            else:
                loss = compute_loss(model(xb), yb, cfg["loss"])
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)               # unscale TRƯỚC khi đo/clip để grad_norm đúng thang
            else:
                loss.backward()
            gn = clip_gradients(model.parameters(), cfg["clip_norm"])      # chuẩn TRƯỚC khi cắt
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            if sched is not None:
                sched.step()
            loss_buf[k], gn_buf[k] = loss.detach(), gn.detach()
        if use_cuda:
            torch.cuda.synchronize()
        t_ep = time.time() - t0

        losses, gns = loss_buf.cpu().numpy(), gn_buf.cpu().numpy()
        all_gn.append(gns)
        tr = evaluate(model, X_sub, y_sub, cfg["loss"])
        va = evaluate(model, X_val, y_val, cfg["loss"])
        for key, v in (("epoch", ep), ("train_loss", tr["loss"]), ("val_loss", va["loss"]),
                       ("val_acc", va["acc"]), ("val_macro_f1", va["macro_f1"]),
                       ("grad_norm", float(_finite_stats(gns).mean())), ("epoch_time_s", t_ep)):
            hist[key].append(v)
        if verbose:
            print(f"  ep{ep:02d} train {tr['loss']:.4f} val {va['loss']:.4f} acc {va['acc']:.4f} "
                  f"f1 {va['macro_f1']:.4f} gn {hist['grad_norm'][-1]:.3f} {t_ep:.1f}s")
        if not (np.isfinite(losses).all() and np.isfinite(va["loss"])):
            diverged = True                                                # dừng sớm, không để notebook treo
            break
        if va["loss"] < best["val_loss"]:
            best.update(val_loss=va["loss"], epoch=ep, acc=va["acc"], f1=va["macro_f1"],
                        state={k_: v_.detach().clone() for k_, v_ in model.state_dict().items()})

    gn_all = _finite_stats(np.concatenate(all_gn))
    nan = float("nan")
    summary = dict(
        step0_loss=step0,
        best_val_loss=best["val_loss"] if best["epoch"] else nan,
        best_epoch=best["epoch"],
        final_train_loss=hist["train_loss"][-1], final_val_loss=hist["val_loss"][-1],
        val_acc=best["acc"] if best["epoch"] else nan,
        val_macro_f1=best["f1"] if best["epoch"] else nan,
        time_per_epoch_s=float(np.mean(hist["epoch_time_s"])),
        peak_mem_MB=torch.cuda.max_memory_allocated() / 2**20 if use_cuda else nan,
        diverged=diverged,
        # thống kê phụ (không có trong bảng xlsx): dùng để chọn ngưỡng clip và giải thích
        gn_p50=float(np.percentile(gn_all, 50)), gn_p95=float(np.percentile(gn_all, 95)),
        gn_max=float(gn_all.max()),
        clip_frac=(float(np.mean(gn_all > cfg["clip_norm"])) if cfg["clip_norm"] else 0.0),
        steps=int(len(np.concatenate(all_gn))), wall_time_s=time.time() - t_run, act_std_step0=act0,
    )
    return dict(cfg=cfg, history=hist, summary=summary, best_state=best["state"])


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi CSV `row_id,pred` (preds là nhãn int64 0..6, cùng thứ tự với row_id)."""
    import pandas as pd
    row_id, preds = np.asarray(row_id).astype(np.int64), np.asarray(preds).astype(np.int64)
    assert len(row_id) == len(preds) == len(set(row_id.tolist())), "row_id phải đủ và không lặp"
    pd.DataFrame({"row_id": row_id, "pred": preds}).to_csv(path, index=False)


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng cho baseline và cấu hình cuối cùng: nạp best_state (epoch val_loss thấp nhất), dự đoán TOÀN BỘ
    eval ở chế độ eval() FP32, ghi predictions. Số điểm do scripts/evaluate.py tính (xem run_evaluate_script)."""
    cfg = {**DEFAULT_CFG, **cfg}
    model = MLP(tuple(cfg["hidden"]), cfg["dropout"], cfg["init"]).to(data["X_eval"].device)
    model.load_state_dict(result["best_state"])
    preds = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)


def run_evaluate_script(pred_path: str, out_json: str, repo_root: str, verbose: bool = True) -> dict:
    """Chạy đúng scripts/evaluate.py (từ repo_root), in kết quả (nếu verbose) và trả về dict từ file JSON."""
    import os
    pred_path, out_json = os.path.abspath(pred_path), os.path.abspath(out_json)
    proc = subprocess.run([sys.executable, "scripts/evaluate.py", "--pred", pred_path, "--out", out_json],
                          cwd=repo_root, capture_output=True, text=True)
    if verbose or proc.returncode:
        print(proc.stdout + proc.stderr)
    proc.check_returncode()
    with open(out_json) as f:
        return json.load(f)
