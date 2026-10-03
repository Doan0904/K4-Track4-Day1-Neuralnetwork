"""plots.py — ảnh biểu đồ: mỗi thí nghiệm một ảnh figures/<exp_id>.png và ảnh chồng theo nhóm."""
from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

OPT_LABEL = {"sgd": "SGD", "sgd_momentum": "SGD+mom", "adam": "Adam", "adamw": "AdamW"}


def cfg_line(cfg: dict) -> str:
    """Một dòng mô tả cấu hình chính cho tiêu đề ảnh."""
    parts = [f"{cfg['loss'].upper()}", f"{OPT_LABEL[cfg['optimizer']]} lr={cfg['lr']:g}",
             f"batch={cfg['batch']}", f"{'-'.join(map(str, cfg['hidden']))}", f"init={cfg['init']}"]
    if cfg.get("weight_decay"):
        parts.append(f"wd={cfg['weight_decay']:g}")
    if cfg.get("dropout"):
        parts.append(f"dropout={cfg['dropout']:g}")
    if cfg.get("clip_norm"):
        parts.append(f"clip={cfg['clip_norm']:g}")
    if cfg.get("precision", "fp32") != "fp32":
        parts.append(cfg["precision"])
    if cfg.get("scheduler"):
        parts.append(cfg["scheduler"])
    parts.append(f"{cfg['epochs']}ep seed={cfg['seed']}")
    return ", ".join(parts)


def plot_run(result: dict, path: str) -> None:
    """Ảnh 3 ô cho MỘT thí nghiệm: (1) train/val loss, (2) val acc + macro-F1, (3) grad_norm (trước clip)."""
    cfg, h, s = result["cfg"], result["history"], result["summary"]
    ep = np.array(h["epoch"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(ep, h["train_loss"], "o-", ms=3, label="train loss (eval mode, 50k mẫu)")
    ax[0].plot(ep, h["val_loss"], "s-", ms=3, label="val loss")
    ax[0].set(xlabel="epoch", ylabel=f"loss ({cfg['loss'].upper()})", title="Train / val loss")
    ax[1].plot(ep, h["val_acc"], "o-", ms=3, label="val accuracy")
    ax[1].plot(ep, h["val_macro_f1"], "s-", ms=3, label="val macro-F1")
    ax[1].set(xlabel="epoch", ylabel="điểm", title="Val accuracy và macro-F1")
    ax[2].plot(ep, h["grad_norm"], "o-", ms=3, color="tab:green", label="grad_norm (TB mỗi epoch)")
    ax[2].set(xlabel="epoch", ylabel="‖g‖₂ toàn cục, trước clip", title="Chuẩn gradient")
    if cfg.get("clip_norm"):
        ax[2].axhline(cfg["clip_norm"], color="r", ls="--", lw=1, label=f"ngưỡng clip c={cfg['clip_norm']:g}")
    if np.all(np.isfinite(h["grad_norm"])) and min(h["grad_norm"]) > 0:
        ax[2].set_yscale("log")
    if s.get("best_epoch"):
        for a in ax:
            a.axvline(s["best_epoch"], color="gray", ls=":", lw=1, label="best epoch" if a is ax[0] else None)
    for a in ax:
        a.grid(alpha=.3)
        a.legend(fontsize=8)
    status = "  [DIVERGED]" if s.get("diverged") else ""
    fig.suptitle(f"{cfg['exp_id']}{status}\n{cfg_line(cfg)}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric, path: str, title: str = "") -> None:
    """Vẽ chồng một (hoặc nhiều) chỉ số của nhiều thí nghiệm; mỗi chỉ số một ô, mỗi thí nghiệm một đường."""
    metrics = [metric] if isinstance(metric, str) else list(metric)
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.2 * len(metrics), 4.2), squeeze=False)
    cmap = matplotlib.colormaps["tab10" if len(results) <= 10 else "tab20"]
    for ax, m in zip(axes[0], metrics):
        allv = []
        for i, r in enumerate(results):
            y = np.array(r["history"][m], dtype=float)
            ax.plot(r["history"]["epoch"], y, "-", marker="o", ms=2.5, lw=1.4, color=cmap(i % cmap.N),
                    label=r["cfg"]["exp_id"])
            allv.append(y[np.isfinite(y)])
        ax.set(xlabel="epoch", ylabel=m, title=m)
        ax.grid(alpha=.3)
        v = np.concatenate(allv) if allv else np.array([])
        if m == "grad_norm" and len(v) and v.min() > 0:
            ax.set_yscale("log")
        elif m in ("val_loss", "train_loss") and len(v):
            ax.set_ylim(v.min() * 0.95, min(v.max(), np.percentile(v, 97)) * 1.05)   # bỏ qua gai phân kỳ
        elif m in ("val_acc", "val_macro_f1") and len(v):
            ax.set_ylim(max(0, np.percentile(v, 3) - 0.02), v.max() + 0.01)
    axes[0][0].legend(fontsize=7, loc="best")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
