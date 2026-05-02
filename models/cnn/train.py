import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import MultiTimeframeCNN

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MASTER_DIR = BASE_DIR / "data" / "master_training"
MODEL_OUTPUT = BASE_DIR / "models" / "pattern_master_cnn.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the multi-timeframe CNN on master training arrays.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_master_dataset() -> TensorDataset:
    print("Loading Master Data...")
    x1 = torch.from_numpy(np.load(MASTER_DIR / "MASTER_X1.npy")).float().unsqueeze(1)
    x5 = torch.from_numpy(np.load(MASTER_DIR / "MASTER_X5.npy")).float().unsqueeze(1)
    xh = torch.from_numpy(np.load(MASTER_DIR / "MASTER_XH.npy")).float().unsqueeze(1)
    y = torch.from_numpy(np.load(MASTER_DIR / "MASTER_y.npy")).long()
    return TensorDataset(x1, x5, xh, y)


def main() -> None:
    args = parse_args()
    dataset = load_master_dataset()
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device(args.device)
    model = MultiTimeframeCNN().to(device)

    weights = torch.tensor([1.0, 10.0, 10.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2, factor=0.5)

    print(f"Starting Training on {device} with {len(dataset)} samples...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for b_x1, b_x5, b_xh, b_y in train_loader:
            b_x1 = b_x1.to(device)
            b_x5 = b_x5.to(device)
            b_xh = b_xh.to(device)
            b_y = b_y.to(device)

            optimizer.zero_grad()
            outputs = model(b_x1, b_x5, b_xh)
            loss = criterion(outputs, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}/{args.epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

    torch.save(model.state_dict(), MODEL_OUTPUT)
    print(f"Model saved to {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
