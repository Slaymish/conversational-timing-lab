"""Train the joint turn-end / response-onset model.

    python -m turntiming.train --config configs/baseline.yaml --data data/synthetic.jsonl
    python -m turntiming.train --config configs/ami.yaml --data data/ami.jsonl

Every run prints the E0 pause-duration baseline on the same validation split,
because a timing model that cannot beat "wait for silence" has not earned its
parameters. Targets are in seconds; only the reports talk in milliseconds.
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from turntiming.baselines.pause import ConstantOnsetBaseline, PauseThresholdBaseline
from turntiming.data.dataset import build_dataset
from turntiming.data.split import group_keys, group_split
from turntiming.eval.metrics import report
from turntiming.models.joint import JointTimingModel


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect(loader: DataLoader, model: torch.nn.Module | None = None) -> dict[str, np.ndarray]:
    """Gather labels -- and, if a model is given, its predictions -- for a split."""
    keys = ("turn_end", "onset_s", "pause_s", "offset_s", "onset_valid")
    buffers: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    buffers["prob"], buffers["onset_pred_s"] = [], []

    for batch in loader:
        for key in keys:
            buffers[key].append(batch[key].numpy())
        if model is not None:
            with torch.no_grad():
                out = model(batch["features"])
            buffers["prob"].append(out["turn_end_logit"].sigmoid().numpy())
            buffers["onset_pred_s"].append(out["onset_delay_s"].numpy())

    return {k: np.concatenate(v) for k, v in buffers.items() if v}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    seed_everything(cfg["seed"])

    dataset = build_dataset(
        args.data,
        window_seconds=cfg.get("window_seconds", 2.0),
        sample_rate=cfg.get("sample_rate", 16_000),
        n_mels=cfg["n_mels"],
    )
    keys = group_keys(dataset.rows, cfg.get("split_by", "conversation"))
    train_idx, val_idx = group_split(keys, cfg.get("val_fraction", 0.2), cfg["seed"])
    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())

    num_workers = cfg.get("num_workers", 0)
    loader_kwargs = {"num_workers": num_workers, "persistent_workers": num_workers > 0}
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, **loader_kwargs)
    train_eval_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], **loader_kwargs)

    model = JointTimingModel(cfg["n_mels"], cfg["hidden_size"])
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    max_fir = cfg.get("max_false_interruption_rate", 0.05)

    print(
        f"split_by={cfg.get('split_by', 'conversation')} train={len(train_ds)} "
        f"val={len(val_ds)} params={sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )

    for epoch in range(cfg["epochs"]):
        model.train()
        for batch in train_loader:
            out = model(batch["features"])
            turn_loss = F.binary_cross_entropy_with_logits(out["turn_end_logit"], batch["turn_end"])

            # Censored onsets say only "further away than the horizon"; don't fit them.
            valid = batch["onset_valid"]
            per_row = F.smooth_l1_loss(out["onset_delay_s"], batch["onset_s"], reduction="none")
            onset_loss = (per_row * valid).sum() / valid.sum().clamp_min(1.0)

            loss = cfg["turn_end_loss_weight"] * turn_loss + cfg["onset_loss_weight"] * onset_loss
            optim.zero_grad()
            loss.backward()
            optim.step()

        model.eval()
        val = collect(val_loader, model)
        summary = report(
            val["turn_end"], val["prob"],
            onset_true_s=val["onset_s"], onset_pred_s=val["onset_pred_s"],
            onset_mask=val["onset_valid"].astype(bool), max_fir=max_fir,
        ).splitlines()
        print(f"epoch={epoch + 1:<3d}{summary[0].strip()}", flush=True)

    train = collect(train_eval_loader)
    pause_model = PauseThresholdBaseline.fit(train["pause_s"], train["turn_end"])
    onset_model = ConstantOnsetBaseline.fit(train["onset_s"][train["onset_valid"].astype(bool)])

    print("\n" + "=" * 78)
    print(
        report(
            val["turn_end"],
            pause_model.decision_score(val["pause_s"]),
            onset_true_s=val["onset_s"],
            onset_pred_s=onset_model.predict(len(val["onset_s"])),
            onset_mask=val["onset_valid"].astype(bool),
            offsets=val["offset_s"],
            max_fir=max_fir,
            title=(
                f"E0 baseline  pause {pause_model.polarity} than "
                f"{pause_model.threshold_s * 1000:.0f}ms, onset always "
                f"{onset_model.onset_s * 1000:.0f}ms"
            ),
        )
    )
    print()
    print(
        report(
            val["turn_end"], val["prob"],
            onset_true_s=val["onset_s"], onset_pred_s=val["onset_pred_s"],
            onset_mask=val["onset_valid"].astype(bool), offsets=val["offset_s"],
            max_fir=max_fir, title="E1 joint audio model",
        )
    )


if __name__ == "__main__":
    main()
