from __future__ import annotations

import argparse
import random

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from turntiming.data.dataset import JsonlTimingDataset
from turntiming.models.joint import JointTimingModel


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seed_everything(cfg["seed"])

    dataset = JsonlTimingDataset(args.data)
    n_val = max(1, int(len(dataset) * 0.2))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"])

    model = JointTimingModel(cfg["n_mels"], cfg["hidden_size"])
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])

    for epoch in range(cfg["epochs"]):
        model.train()
        for batch in train_loader:
            out = model(batch["features"])
            turn_loss = F.binary_cross_entropy_with_logits(out["turn_end_logit"], batch["turn_end"])
            onset_loss = F.smooth_l1_loss(out["onset_delay_ms"], batch["onset_ms"])
            loss = (
                cfg["turn_end_loss_weight"] * turn_loss
                + cfg["onset_loss_weight"] * onset_loss / 1000.0
            )
            optim.zero_grad()
            loss.backward()
            optim.step()

        model.eval()
        correct = total = 0
        onset_abs_error = 0.0
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch["features"])
                pred = (out["turn_end_logit"].sigmoid() >= 0.5).float()
                correct += int((pred == batch["turn_end"]).sum())
                total += len(pred)
                onset_abs_error += float((out["onset_delay_ms"] - batch["onset_ms"]).abs().sum())
        print(
            f"epoch={epoch + 1} "
            f"turn_acc={correct / max(total, 1):.3f} "
            f"onset_mae_ms={onset_abs_error / max(total, 1):.1f}"
        )


if __name__ == "__main__":
    main()
