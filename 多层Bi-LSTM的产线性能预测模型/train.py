import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from generate_data import FEATURE_COLUMNS, TARGET_COLUMNS
from model import ManufacturingDBiLSTM


def make_windows(features, targets, sequence_length, horizon=1):
    xs, ys = [], []
    last_start = len(features) - sequence_length - horizon + 1

    for start in range(last_start):
        end = start + sequence_length
        xs.append(features[start:end])
        ys.append(targets[end + horizon - 1])

    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
    )


def calculate_metrics(y_true, y_pred):
    rows = []
    for index, name in enumerate(TARGET_COLUMNS):
        true = y_true[:, index]
        pred = y_pred[:, index]

        rows.append({
            "target": name,
            "RMSE": np.sqrt(mean_squared_error(true, pred)),
            "MAE": mean_absolute_error(true, pred),
            "MAPE": np.mean(
                np.abs((true - pred) / np.maximum(np.abs(true), 1e-6))
            ) * 100,
            "R2": r2_score(true, pred),
        })

    return pd.DataFrame(rows)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(features)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        loss = criterion(model(features), targets)
        total_loss += loss.item() * len(features)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    outputs = []

    for features, _ in loader:
        outputs.append(model(features.to(device)).cpu().numpy())

    return np.concatenate(outputs)


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    frame = pd.read_csv(args.data)
    feature_values = frame[FEATURE_COLUMNS].to_numpy()
    target_values = frame[TARGET_COLUMNS].to_numpy()

    n = len(frame)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    # Scalers only learn from the training period.
    feature_scaler = StandardScaler().fit(
        feature_values[:train_end]
    )
    target_scaler = StandardScaler().fit(
        target_values[:train_end]
    )

    scaled_features = feature_scaler.transform(feature_values)
    scaled_targets = target_scaler.transform(target_values)

    x, y = make_windows(
        scaled_features,
        scaled_targets,
        args.sequence_length,
        args.horizon,
    )

    # Window target index equals start + sequence_length + horizon - 1.
    target_indices = (
        np.arange(len(x))
        + args.sequence_length
        + args.horizon
        - 1
    )

    train_mask = target_indices < train_end
    val_mask = (
        (target_indices >= train_end)
        & (target_indices < val_end)
    )
    test_mask = target_indices >= val_end

    train_set = TensorDataset(
        torch.from_numpy(x[train_mask]),
        torch.from_numpy(y[train_mask]),
    )
    val_set = TensorDataset(
        torch.from_numpy(x[val_mask]),
        torch.from_numpy(y[val_mask]),
    )
    test_set = TensorDataset(
        torch.from_numpy(x[test_mask]),
        torch.from_numpy(y[test_mask]),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = ManufacturingDBiLSTM(
        input_size=len(FEATURE_COLUMNS),
        output_size=len(TARGET_COLUMNS),
        conv_channels=args.conv_channels,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        attention_heads=args.attention_heads,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model.pt"

    best_val = float("inf")
    wait = 0
    train_history, val_history = [], []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )
        val_loss = evaluate_loss(
            model,
            val_loader,
            criterion,
            device,
        )
        scheduler.step(val_loss)

        train_history.append(train_loss)
        val_history.append(val_loss)

        print(
            f"Epoch {epoch:03d} "
            f"train={train_loss:.6f} val={val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), best_path)
        else:
            wait += 1
            if wait >= args.patience:
                print("Early stopping")
                break

    model.load_state_dict(
        torch.load(best_path, map_location=device)
    )

    scaled_predictions = predict(model, test_loader, device)
    scaled_truth = y[test_mask]

    predictions = target_scaler.inverse_transform(
        scaled_predictions
    )
    truth = target_scaler.inverse_transform(scaled_truth)

    metrics = calculate_metrics(truth, predictions)
    print(metrics.to_string(index=False))
    metrics.to_csv(output_dir / "metrics.csv", index=False)

    result = pd.DataFrame()
    for index, target in enumerate(TARGET_COLUMNS):
        result[f"{target}_actual"] = truth[:, index]
        result[f"{target}_predicted"] = predictions[:, index]
    result.to_csv(output_dir / "predictions.csv", index=False)

    joblib.dump(
        feature_scaler,
        output_dir / "feature_scaler.joblib",
    )
    joblib.dump(
        target_scaler,
        output_dir / "target_scaler.joblib",
    )

    plt.figure(figsize=(11, 5))
    count = min(500, len(truth))
    plt.plot(
        truth[:count, 0],
        label="Actual production time",
    )
    plt.plot(
        predictions[:count, 0],
        label="Predicted production time",
    )
    plt.xlabel("Test sample")
    plt.ylabel("Production time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(train_history, label="Train")
    plt.plot(val_history, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Smooth L1 loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "learning_curve.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/manufacturing.csv",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--conv-channels", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
