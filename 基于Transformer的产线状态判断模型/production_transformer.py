import argparse
import json
import math
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


STATE_NAMES = {
    "process": ["正常", "工艺漂移", "温度异常", "质量风险"],
    "equipment": ["正常", "磨损", "过载", "故障"],
    "logistics": ["正常", "缺料", "堵塞", "AGV延迟"],
    "line": ["正常", "低效", "瓶颈", "停线"],
}


FEATURES = [
    "temperature",
    "pressure",
    "flow_rate",
    "spindle_speed",
    "feed_rate",
    "vibration",
    "motor_current",
    "motor_voltage",
    "bearing_temperature",
    "tool_wear",
    "equipment_load",
    "cycle_time",
    "buffer_level",
    "material_inventory",
    "conveyor_speed",
    "agv_delay",
    "queue_length",
    "station_output",
    "defect_rate",
    "energy_consumption",
    "oee",
    "line_takt",
    "downtime",
    "throughput",
]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_args(args):
    if args.samples <= 0:
        raise ValueError("--samples 必须大于 0")

    if args.seq_len < 2:
        raise ValueError("--seq-len 必须至少为 2")

    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须大于 0")

    if args.epochs <= 0:
        raise ValueError("--epochs 必须大于 0")

    if args.d_model <= 0:
        raise ValueError("--d-model 必须大于 0")

    if args.heads <= 0:
        raise ValueError("--heads 必须大于 0")

    if args.d_model % args.heads != 0:
        raise ValueError(
            f"--d-model 必须能被 --heads 整除，当前为 "
            f"d_model={args.d_model}, heads={args.heads}"
        )

    if args.lr <= 0:
        raise ValueError("--lr 必须大于 0")

    if args.samples < args.seq_len * 3:
        raise ValueError(
            "--samples 太小，至少应为 --seq-len 的 3 倍，"
            "以便划分训练集、验证集和测试集"
        )


def markov_states(n, probabilities, persistence=0.96):
    states = np.zeros(n, dtype=np.int64)

    for i in range(1, n):
        if np.random.rand() < persistence:
            states[i] = states[i - 1]
        else:
            states[i] = np.random.choice(
                len(probabilities),
                p=probabilities,
            )

    return states


def simulate_data(n=30000, seed=42):
    set_seed(seed)

    t = np.arange(n)

    process = markov_states(
        n,
        [0.64, 0.14, 0.12, 0.10],
    )

    equipment = markov_states(
        n,
        [0.66, 0.14, 0.12, 0.08],
    )

    logistics = markov_states(
        n,
        [0.68, 0.12, 0.12, 0.08],
    )

    process_effect = np.array([
        [0, 0, 0, 0, 0],
        [4, 0.35, -2, -20, 0.008],
        [18, 0.15, -4, -40, 0.015],
        [7, -0.20, -6, -60, 0.055],
    ])[process]

    equipment_effect = np.array([
        [0, 0, 0, 0, 0],
        [1.3, 7, 5, 0.08, 3],
        [2.0, 15, 12, 0.12, 7],
        [4.5, 25, 28, 0.25, 20],
    ])[equipment]

    logistics_effect = np.array([
        [0, 0, 0, 0],
        [-22, 2, -3, 3],
        [25, 7, -7, 8],
        [-12, 5, -5, 10],
    ])[logistics]

    line = np.zeros(n, dtype=np.int64)

    line[
        (process > 0)
        | (equipment == 1)
        | (logistics == 1)
    ] = 1

    line[
        (equipment == 2)
        | (logistics == 2)
        | (process == 3)
    ] = 2

    line[
        (equipment == 3)
        | ((logistics == 3) & (process > 0))
    ] = 3

    def noise(scale):
        return np.random.normal(0, scale, n)

    cycle = np.sin(2 * np.pi * t / 1440)

    df = pd.DataFrame({
        "timestamp": pd.date_range(
            "2025-01-01",
            periods=n,
            freq="s",
        ),
        "temperature": (
            65
            + 2 * cycle
            + process_effect[:, 0]
            + noise(1.0)
        ),
        "pressure": (
            5
            + process_effect[:, 1]
            + noise(0.10)
        ),
        "flow_rate": (
            80
            + process_effect[:, 2]
            + noise(1.5)
        ),
        "spindle_speed": (
            1500
            + process_effect[:, 3]
            + noise(15)
        ),
        "feed_rate": (
            300
            - 300 * process_effect[:, 4]
            + noise(4)
        ),
        "vibration": (
            1.2
            + equipment_effect[:, 0]
            + noise(0.12)
        ),
        "motor_current": (
            35
            + equipment_effect[:, 1]
            + noise(1.2)
        ),
        "motor_voltage": 380 + noise(2),
        "bearing_temperature": (
            55
            + equipment_effect[:, 2]
            + noise(1.0)
        ),
        "tool_wear": np.clip(
            0.1
            + t / n * 0.5
            + equipment_effect[:, 3]
            + noise(0.02),
            0,
            1,
        ),
        "equipment_load": (
            70
            + equipment_effect[:, 4]
            + noise(2)
        ),
        "cycle_time": (
            12
            + line * 1.8
            + noise(0.3)
        ),
        "buffer_level": np.clip(
            50
            + logistics_effect[:, 0]
            + noise(3),
            0,
            100,
        ),
        "material_inventory": np.clip(
            70
            - 18 * (logistics == 1)
            + noise(4),
            0,
            100,
        ),
        "conveyor_speed": np.clip(
            1.5
            - 0.15 * logistics
            + noise(0.04),
            0,
            None,
        ),
        "agv_delay": np.clip(
            logistics_effect[:, 1]
            + noise(0.4),
            0,
            None,
        ),
        "queue_length": np.clip(
            3
            + logistics_effect[:, 2]
            + noise(1),
            0,
            None,
        ),
        "station_output": np.clip(
            50
            + logistics_effect[:, 3]
            - 6 * line
            + noise(2),
            0,
            None,
        ),
        "defect_rate": np.clip(
            0.01
            + process_effect[:, 4]
            + noise(0.003),
            0,
            1,
        ),
        "energy_consumption": (
            120
            + equipment_effect[:, 1]
            + 5 * line
            + noise(3)
        ),
        "oee": np.clip(
            0.92
            - 0.12 * line
            - 0.03 * process
            + noise(0.015),
            0,
            1,
        ),
        "line_takt": (
            12
            + 1.5 * line
            + noise(0.3)
        ),
        "downtime": np.clip(
            15 * (line == 3)
            + noise(0.5),
            0,
            None,
        ),
        "throughput": np.clip(
            60
            - 10 * line
            - logistics_effect[:, 3]
            + noise(2),
            0,
            None,
        ),
        "process_label": process,
        "equipment_label": equipment,
        "logistics_label": logistics,
        "line_label": line,
    })

    return df


def make_parameter_table(df):
    rows = []

    for task in STATE_NAMES:
        label_col = f"{task}_label"

        for label, name in enumerate(STATE_NAMES[task]):
            part = df[df[label_col] == label]

            for feature in FEATURES:
                rows.append({
                    "state_category": task,
                    "state_code": label,
                    "state_name": name,
                    "parameter": feature,
                    "mean": part[feature].mean(),
                    "std": part[feature].std(),
                    "min": part[feature].min(),
                    "max": part[feature].max(),
                    "p05": part[feature].quantile(0.05),
                    "p95": part[feature].quantile(0.95),
                })

    return pd.DataFrame(rows)


class SequenceDataset(Dataset):
    def __init__(self, values, labels, seq_len, indices):
        self.values = torch.tensor(
            values,
            dtype=torch.float32,
        )
        self.labels = labels
        self.seq_len = seq_len
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        end = int(self.indices[index])
        start = end - self.seq_len + 1

        x = self.values[start:end + 1]

        y = {
            task: torch.tensor(
                labels[end],
                dtype=torch.long,
            )
            for task, labels in self.labels.items()
        }

        return x, y


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)

        divisor = torch.exp(
            torch.arange(0, d_model, 2)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * divisor)

        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * divisor)
        else:
            pe[:, 1::2] = torch.cos(
                position * divisor[:-1]
            )

        self.register_buffer(
            "pe",
            pe.unsqueeze(0),
        )

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"序列长度 {x.size(1)} 超过最大位置编码长度 "
                f"{self.pe.size(1)}"
            )

        return x + self.pe[:, :x.size(1)]


class ProductionTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=128,
        nhead=8,
        layers=3,
        dropout=0.1,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model={d_model} 必须能被 nhead={nhead} 整除"
            )

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.position = PositionalEncoding(
            d_model,
            max_len=2048,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            layers,
        )

        self.norm = nn.LayerNorm(d_model)

        self.heads = nn.ModuleDict({
            task: nn.Linear(d_model, len(names))
            for task, names in STATE_NAMES.items()
        })

    def forward(self, x):
        x = self.input_projection(x)
        x = self.position(x)

        encoded = self.encoder(x)
        encoded = self.norm(encoded)

        # 最近时刻通常最能反映当前状态，避免全序列平均导致异常信号被稀释。
        pooled = encoded[:, -1]

        return {
            task: head(pooled)
            for task, head in self.heads.items()
        }


def calculate_class_weights(labels, indices, task):
    target = labels[task][indices]
    class_count = len(STATE_NAMES[task])

    counts = np.bincount(
        target,
        minlength=class_count,
    ).astype(np.float32)

    counts = np.maximum(counts, 1.0)

    weights = counts.sum() / (class_count * counts)
    weights = weights / weights.mean()

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


def evaluate(model, loader, device, detailed=False):
    model.eval()

    correct = {
        task: 0
        for task in STATE_NAMES
    }

    total = 0

    predictions = {
        task: []
        for task in STATE_NAMES
    }

    targets = {
        task: []
        for task in STATE_NAMES
    }

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            outputs = model(x)

            total += x.size(0)

            for task in STATE_NAMES:
                pred = outputs[task].argmax(1).cpu()
                target = y[task]

                correct[task] += (
                    pred == target
                ).sum().item()

                predictions[task].extend(
                    pred.tolist()
                )

                targets[task].extend(
                    target.tolist()
                )

    metrics = {
        task: correct[task] / max(total, 1)
        for task in STATE_NAMES
    }

    if detailed:
        for task in STATE_NAMES:
            print(f"\n[{task}]")
            print(
                classification_report(
                    targets[task],
                    predictions[task],
                    labels=range(len(STATE_NAMES[task])),
                    target_names=STATE_NAMES[task],
                    zero_division=0,
                )
            )

    return metrics


def build_loader(values, labels, seq_len, indices, batch_size, shuffle):
    dataset = SequenceDataset(
        values=values,
        labels=labels,
        seq_len=seq_len,
        indices=indices,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )


def train(args):
    validate_args(args)
    set_seed(args.seed)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    df = simulate_data(
        n=args.samples,
        seed=args.seed,
    )

    df.to_csv(
        output / "simulation_data.csv",
        index=False,
    )

    make_parameter_table(df).to_csv(
        output / "production_state_parameters.csv",
        index=False,
    )

    total_samples = len(df)

    train_end = int(total_samples * 0.70)
    valid_end = int(total_samples * 0.85)

    scaler = StandardScaler()

    values = np.empty(
        (total_samples, len(FEATURES)),
        dtype=np.float32,
    )

    # 只使用训练区间拟合 scaler，避免验证集和测试集信息泄漏。
    values[:train_end] = scaler.fit_transform(
        df.iloc[:train_end][FEATURES]
    )

    values[train_end:] = scaler.transform(
        df.iloc[train_end:][FEATURES]
    )

    joblib.dump(
        scaler,
        output / "scaler.joblib",
    )

    labels = {
        task: df[f"{task}_label"].to_numpy(
            dtype=np.int64
        )
        for task in STATE_NAMES
    }

    # 每个区间只使用该区间内部完整的窗口，避免窗口跨越数据集边界。
    train_indices = np.arange(
        args.seq_len - 1,
        train_end,
    )

    valid_indices = np.arange(
        train_end + args.seq_len - 1,
        valid_end,
    )

    test_indices = np.arange(
        valid_end + args.seq_len - 1,
        total_samples,
    )

    if len(train_indices) == 0:
        raise ValueError("训练集无法构造时序窗口")

    if len(valid_indices) == 0:
        raise ValueError("验证集无法构造时序窗口")

    if len(test_indices) == 0:
        raise ValueError("测试集无法构造时序窗口")

    train_loader = build_loader(
        values,
        labels,
        args.seq_len,
        train_indices,
        args.batch_size,
        shuffle=True,
    )

    valid_loader = build_loader(
        values,
        labels,
        args.seq_len,
        valid_indices,
        args.batch_size,
        shuffle=False,
    )

    test_loader = build_loader(
        values,
        labels,
        args.seq_len,
        test_indices,
        args.batch_size,
        shuffle=False,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = ProductionTransformer(
        input_dim=len(FEATURES),
        d_model=args.d_model,
        nhead=args.heads,
        layers=args.layers,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    criteria = {}

    for task in STATE_NAMES:
        class_weights = calculate_class_weights(
            labels,
            train_indices,
            task,
        ).to(device)

        criteria[task] = nn.CrossEntropyLoss(
            weight=class_weights
        )

    best_score = -float("inf")
    best_epoch = 0
    best_state = None
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)

            outputs = model(x)

            loss = sum(
                criteria[task](
                    outputs[task],
                    y[task].to(device),
                )
                for task in STATE_NAMES
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            running_loss += loss.item()

        valid_metrics = evaluate(
            model,
            valid_loader,
            device,
        )

        valid_score = float(
            np.mean(list(valid_metrics.values()))
        )

        scores = " ".join(
            f"{task}={score:.3f}"
            for task, score in valid_metrics.items()
        )

        average_loss = running_loss / max(
            len(train_loader),
            1,
        )

        print(
            f"epoch={epoch} "
            f"loss={average_loss:.4f} "
            f"val_mean={valid_score:.3f} "
            f"{scores}"
        )

        if valid_score > best_score:
            best_score = valid_score
            best_epoch = epoch
            patience_count = 0

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            patience_count += 1

        if patience_count >= args.patience:
            print(
                f"验证集连续 {args.patience} 个 epoch 未提升，提前停止"
            )
            break

    if best_state is None:
        raise RuntimeError("训练未生成有效模型参数")

    model.load_state_dict(best_state)

    print(
        f"\n恢复最佳模型：epoch={best_epoch}, "
        f"validation_mean_accuracy={best_score:.4f}"
    )

    test_metrics = evaluate(
        model,
        test_loader,
        device,
        detailed=True,
    )

    print(
        "\n[test] "
        + " ".join(
            f"{task}={score:.3f}"
            for task, score in test_metrics.items()
        )
    )

    torch.save(
        model.state_dict(),
        output / "transformer.pt",
    )

    config = {
        "features": FEATURES,
        "state_names": STATE_NAMES,
        "seq_len": args.seq_len,
        "d_model": args.d_model,
        "heads": args.heads,
        "layers": args.layers,
        "best_epoch": best_epoch,
        "validation_mean_accuracy": best_score,
    }

    (output / "config.json").write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def predict(csv_path, model_dir):
    model_dir = Path(model_dir)

    config_path = model_dir / "config.json"
    scaler_path = model_dir / "scaler.joblib"
    model_path = model_dir / "transformer.pt"

    for path in [config_path, scaler_path, model_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"缺少模型文件：{path}"
            )

    config = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    df = pd.read_csv(csv_path)

    missing = sorted(
        set(config["features"]) - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"输入数据缺少字段: {missing}"
        )

    if len(df) < config["seq_len"]:
        raise ValueError(
            f"至少需要 {config['seq_len']} 条连续时序记录"
        )

    feature_df = df[config["features"]].tail(
        config["seq_len"]
    )

    if feature_df.isnull().any().any():
        missing_values = feature_df.columns[
            feature_df.isnull().any()
        ].tolist()

        raise ValueError(
            f"输入数据包含缺失值: {missing_values}"
        )

    for feature in config["features"]:
        feature_df[feature] = pd.to_numeric(
            feature_df[feature],
            errors="raise",
        )

    scaler = joblib.load(scaler_path)

    scaled = scaler.transform(feature_df)
    x = torch.tensor(
        scaled,
        dtype=torch.float32,
    ).unsqueeze(0)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = ProductionTransformer(
        input_dim=len(config["features"]),
        d_model=config["d_model"],
        nhead=config["heads"],
        layers=config["layers"],
    ).to(device)

    state_dict = torch.load(
        model_path,
        map_location=device,
    )

    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        outputs = model(x.to(device))

        result = {}

        for task, logits in outputs.items():
            probabilities = torch.softmax(
                logits,
                dim=1,
            )[0].cpu().numpy()

            code = int(
                probabilities.argmax()
            )

            result[task] = {
                "code": code,
                "name": config["state_names"][task][code],
                "confidence": round(
                    float(probabilities[code]),
                    6,
                ),
                "probabilities": {
                    name: round(float(probability), 6)
                    for name, probability in zip(
                        config["state_names"][task],
                        probabilities,
                    )
                },
            }

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    train_parser = sub.add_parser("train")
    train_parser.add_argument(
        "--samples",
        type=int,
        default=30000,
    )
    train_parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=15,
    )
    train_parser.add_argument(
        "--patience",
        type=int,
        default=4,
    )
    train_parser.add_argument(
        "--d-model",
        type=int,
        default=128,
    )
    train_parser.add_argument(
        "--heads",
        type=int,
        default=8,
    )
    train_parser.add_argument(
        "--layers",
        type=int,
        default=3,
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    train_parser.add_argument(
        "--output",
        default="output",
    )

    predict_parser = sub.add_parser("predict")
    predict_parser.add_argument(
        "--csv",
        required=True,
    )
    predict_parser.add_argument(
        "--model-dir",
        default="output",
    )

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    else:
        predict(
            args.csv,
            args.model_dir,
        )


if __name__ == "__main__":
    main()
