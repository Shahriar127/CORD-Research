"""Cross-fitted training of one fold model ``f_{-j}`` on ``D \\ D_j``.

Features:

* data and augmentation on the accelerator (see :mod:`cord.training.augment`);
* mixed precision (bf16 where supported, else fp16 with loss scaling), channels-last, optional
  ``torch.compile``;
* the *confidence trajectory* signal is recorded during training: every ``trajectory_every``
  epochs the softmax probability of the given label is stored for every scored point
  (``D_j ∪ C``, never trained on), so no checkpoint has to be reloaded later;
* resumable: full state (model, optimiser, scaler, RNG, trajectory) is saved atomically every
  ``checkpoint_every`` epochs, so a killed notebook session continues where it stopped.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from cord.config import TrainConfig
from cord.training.augment import as_float, iterate_batches, random_crop_flip, to_device_images
from cord.utils.io import append_jsonl, save_npz, save_torch, write_json
from cord.utils.runtime import get_logger

log = get_logger("train")


@dataclass(frozen=True)
class FoldData:
    """Arrays for one fold. Images are uint8 NHWC."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_scored: np.ndarray  # D_j ∪ C, in Split.scored_indices(j) order
    y_scored: np.ndarray  # given (possibly poisoned) labels of the scored points
    x_test: np.ndarray
    y_test: np.ndarray
    x_backdoor: np.ndarray  # triggered test inputs (empty for the no-attack condition)
    y_backdoor: np.ndarray


# ---------------------------------------------------------------------------------------------
# precision helpers
# ---------------------------------------------------------------------------------------------


def native_bf16(device: str) -> bool:
    """True on GPUs with hardware bf16 (compute capability >= 8.0: A100, L4, H100, RTX 30xx+).

    ``torch.cuda.is_bf16_supported()`` also returns True when bf16 is only *emulated* (e.g. Tesla
    T4, P100), where it is much slower than fp16; so the capability is checked directly.
    """
    if not device.startswith("cuda"):
        return False
    index = torch.device(device).index
    major, _ = torch.cuda.get_device_capability(index if index is not None else torch.cuda.current_device())
    return major >= 8


def autocast_context(device: str, enabled: bool) -> AbstractContextManager[Any]:
    """bf16 autocast on Ampere and newer, fp16 elsewhere (with loss scaling, see below)."""
    if not enabled or not device.startswith("cuda"):
        return nullcontext()
    dtype = torch.bfloat16 if native_bf16(device) else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def needs_grad_scaler(device: str, enabled: bool) -> bool:
    return enabled and device.startswith("cuda") and not native_bf16(device)


@torch.no_grad()
def predict(
    model: nn.Module,
    x: Tensor,
    batch_size: int,
    device: str,
    amp: bool,
    fn: Callable[[nn.Module, Tensor], Any] | None = None,
) -> Any:
    """Run ``fn(model, batch)`` (default: logits) over a uint8 NCHW tensor; concatenates outputs.

    ``fn`` may return a tensor or a dict of tensors; outputs are moved to CPU float32.
    """
    model.eval()
    fn = fn or (lambda m, b: m(b))
    outs: list[Any] = []
    for start in range(0, x.shape[0], batch_size):
        batch = as_float(x[start : start + batch_size])
        with autocast_context(device, amp):
            out = fn(model, batch)
        if isinstance(out, dict):
            outs.append({k: v.float().cpu() for k, v in out.items()})
        else:
            outs.append(out.float().cpu())
    if not outs:
        return torch.empty(0)
    if isinstance(outs[0], dict):
        return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}
    return torch.cat(outs)


def accuracy(model: nn.Module, x: Tensor, y: Tensor, batch_size: int, device: str, amp: bool) -> float:
    if x.shape[0] == 0:
        return float("nan")
    logits = predict(model, x, batch_size, device, amp)
    return float((logits.argmax(1) == y.cpu()).float().mean())


# ---------------------------------------------------------------------------------------------


def _lr_factor(step: int, total: int, warmup: int, schedule: str) -> float:
    if warmup and step < warmup:
        return (step + 1) / warmup
    if schedule == "constant":
        return 1.0
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def train_fold_model(
    model: nn.Module,
    data: FoldData,
    cfg: TrainConfig,
    device: str,
    seed: int,
    work_dir: Path,
) -> dict[str, Any]:
    """Train (or resume) one fold model; writes model.pt, trajectory.npz, train_summary.json."""
    work_dir.mkdir(parents=True, exist_ok=True)
    state_path, log_path = work_dir / "state.pt", work_dir / "train_log.jsonl"

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    x_tr = to_device_images(data.x_train, device)
    y_tr = torch.from_numpy(data.y_train).long().to(device)
    x_sc = to_device_images(data.x_scored, device)
    y_sc = torch.from_numpy(data.y_scored).long()
    x_te, y_te = to_device_images(data.x_test, device), torch.from_numpy(data.y_test).long()
    x_bd, y_bd = to_device_images(data.x_backdoor, device), torch.from_numpy(data.y_backdoor).long()

    model = model.to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
    run_model: nn.Module = torch.compile(model) if cfg.compile else model  # type: ignore[assignment]

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    opt = torch.optim.SGD(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr,
        momentum=cfg.momentum,
        nesterov=cfg.nesterov,
    )
    steps_per_epoch = math.ceil(len(y_tr) / cfg.batch_size)
    total_steps = cfg.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: _lr_factor(s, total_steps, cfg.warmup_epochs * steps_per_epoch, cfg.schedule)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=needs_grad_scaler(device, cfg.amp))

    start_epoch = 0
    traj_epochs: list[int] = []
    traj_p: list[np.ndarray] = []
    traj_pred: list[np.ndarray] = []
    if state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        gen.set_state(state["generator"].cpu())  # generator states are CPU byte tensors
        start_epoch = state["epoch"]
        traj_epochs, traj_p, traj_pred = state["traj_epochs"], state["traj_p"], state["traj_pred"]
        log.info("resumed from epoch %d", start_epoch)

    def record_trajectory(epoch: int) -> None:
        logits = predict(run_model, x_sc, cfg.eval_batch_size, device, cfg.amp)
        probs = logits.softmax(1)
        traj_epochs.append(epoch)
        traj_p.append(probs.gather(1, y_sc[:, None]).squeeze(1).numpy().astype(np.float16))
        traj_pred.append(logits.argmax(1).numpy().astype(np.int16))

    t0 = time.perf_counter()
    for epoch in range(start_epoch, cfg.epochs):
        run_model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        for idx in iterate_batches(len(y_tr), cfg.batch_size, True, gen, device):
            xb = random_crop_flip(as_float(x_tr[idx]), cfg.crop_padding, gen, cfg.hflip)
            if cfg.channels_last:
                xb = xb.contiguous(memory_format=torch.channels_last)
            yb = y_tr[idx]
            with autocast_context(device, cfg.amp):
                logits = run_model(xb)
                loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += float(loss.detach()) * len(idx)
            correct += int((logits.argmax(1) == yb).sum())
            seen += len(idx)

        ep = epoch + 1
        record = {
            "epoch": ep,
            "loss": loss_sum / seen,
            "train_acc": correct / seen,
            "lr": sched.get_last_lr()[0],
            "elapsed_s": time.perf_counter() - t0,
        }
        if ep == 1 or ep % cfg.trajectory_every == 0 or ep == cfg.epochs:
            record_trajectory(ep)
            record["clean_acc"] = accuracy(run_model, x_te, y_te, cfg.eval_batch_size, device, cfg.amp)
            record["asr"] = accuracy(run_model, x_bd, y_bd, cfg.eval_batch_size, device, cfg.amp)
            log.info(
                "epoch %d/%d loss %.4f train %.3f clean %.4f asr %.4f",
                ep,
                cfg.epochs,
                record["loss"],
                record["train_acc"],
                record["clean_acc"],
                record["asr"],
            )
        append_jsonl(log_path, record)

        if ep % cfg.checkpoint_every == 0 and ep < cfg.epochs:
            save_torch(
                state_path,
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict(),
                    "generator": gen.get_state(),
                    "epoch": ep,
                    "traj_epochs": traj_epochs,
                    "traj_p": traj_p,
                    "traj_pred": traj_pred,
                },
            )

    summary = {
        "clean_acc": accuracy(run_model, x_te, y_te, cfg.eval_batch_size, device, cfg.amp),
        "asr": accuracy(run_model, x_bd, y_bd, cfg.eval_batch_size, device, cfg.amp),
        "n_train": len(y_tr),
        "n_scored": len(y_sc),
        "train_seconds_this_session": time.perf_counter() - t0,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "precision": ("bf16" if native_bf16(device) else "fp16") if cfg.amp and device.startswith("cuda") else "fp32",
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device.startswith("cuda") else None,
    }
    save_torch(work_dir / "model.pt", model.state_dict())
    save_npz(
        work_dir / "trajectory.npz",
        epochs=np.asarray(traj_epochs, np.int32),
        p_given=np.stack(traj_p),
        pred=np.stack(traj_pred),
    )
    write_json(work_dir / "train_summary.json", summary)
    state_path.unlink(missing_ok=True)
    log.info("fold done: clean acc %.4f, ASR %.4f", summary["clean_acc"], summary["asr"])
    return summary
