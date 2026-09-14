from __future__ import annotations

import argparse
import importlib.util
import math
import random
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REQUIRED_PACKAGES = ["torch", "matplotlib", "numpy"]
missing_packages = [name for name in REQUIRED_PACKAGES if importlib.util.find_spec(name) is None]
if missing_packages:
    missing_display = ", ".join(missing_packages)
    raise RuntimeError(
        f"Missing packages: {missing_display}. Install them with `uv add torch matplotlib`."
    )


warnings.filterwarnings("ignore", message="enable_nested_tensor is True")
plt.style.use("seaborn-v0_8-whitegrid")
torch.set_float32_matmul_precision("high")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    else "cpu"
)


@dataclass(frozen=True)
class ExperimentConfig:
    prime: int = 97
    operation: str = "div"
    train_fraction: float = 0.5
    split_seed: int = 1
    model_seed: int = 1
    batch_size: int = 512
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    mlp_mult: int = 4
    lr: float = 1e-3
    weight_decay: float = 0.0
    warmup_steps: int = 10
    max_steps: int = 1_000_000
    eval_every: int = 1_000


OPERATIONS: dict[str, Callable[[int, int, int], int]] = {
    "add": lambda x, y, p: (x + y) % p,
    "sub": lambda x, y, p: (x - y) % p,
    "mul": lambda x, y, p: (x * y) % p,
    "div": lambda x, y, p: (x * pow(y, -1, p)) % p,
    "poly2": lambda x, y, p: (x * x + x * y + y * y) % p,
}

OP_SYMBOLS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "poly2": "f",
}

BASE_CONFIG = ExperimentConfig()

# The released paper code used one vocabulary shared by every task in the
# paper, including S5. Keeping that output dimension is important: using only
# the 97 answer classes makes the optimization problem noticeably easier.
PAPER_OPERATORS = sorted(
    [
        "+", "-", "*", "/", "**2+", "**3+", "+*", "+-",
        "(x._value//y)if(y._value%2==1)else(x-y)_mod_97",
        "copy", "reverse", "s5", "s5aba", "s5conj", "sort",
        "x**2+y**2_mod_97", "x**2+y**2+x*y_mod_97",
        "x**2+y**2+x*y+x_mod_97", "x**3+x*y_mod_97",
        "x**3+x*y**2+y_mod_97",
    ]
)
PAPER_VOCAB_SIZE = 2 + len(PAPER_OPERATORS) + 97 + math.factorial(5)
PAPER_EOS_TOKEN = 0
PAPER_EQ_TOKEN = 1
PAPER_OPERATOR_TOKEN = 2 + PAPER_OPERATORS.index("/")
PAPER_NUMBER_OFFSET = 2 + len(PAPER_OPERATORS)


def build_modular_dataset(config: ExperimentConfig) -> tuple[torch.Tensor, torch.Tensor]:
    if config.operation not in OPERATIONS:
        raise ValueError(f"Unsupported operation: {config.operation}")

    op_token = config.prime
    eq_token = config.prime + 1
    y_values = range(1, config.prime) if config.operation == "div" else range(config.prime)

    inputs: list[list[int]] = []
    targets: list[int] = []

    for x in range(config.prime):
        for y in y_values:
            target = OPERATIONS[config.operation](x, y, config.prime)
            inputs.append([x, op_token, y, eq_token])
            targets.append(target)

    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def build_figure1_dataset(config: ExperimentConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize division exactly like the authors' released implementation.

    Input is ``<eos> x / y = answer`` and the two supervised next-token
    targets are ``answer <eos>``. The paper describes this as calculating loss
    only on the answer (right-hand-side) portion of the equation.
    """
    if config.prime != 97 or config.operation != "div":
        raise ValueError("The Figure 1 tokenizer is defined for division mod 97")

    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    for x in range(97):
        for y in range(1, 97):
            answer = OPERATIONS["div"](x, y, 97)
            inputs.append(
                [
                    PAPER_EOS_TOKEN,
                    PAPER_NUMBER_OFFSET + x,
                    PAPER_OPERATOR_TOKEN,
                    PAPER_NUMBER_OFFSET + y,
                    PAPER_EQ_TOKEN,
                    PAPER_NUMBER_OFFSET + answer,
                ]
            )
            targets.append([PAPER_NUMBER_OFFSET + answer, PAPER_EOS_TOKEN])

    # ArithmeticDataset.make_data(..., seed=0) performs this shuffle before
    # taking the first train_fraction of equations.
    permutation = np.random.RandomState(seed=0).permutation(len(inputs))
    return (
        torch.tensor(np.asarray(inputs)[permutation], dtype=torch.long),
        torch.tensor(np.asarray(targets)[permutation], dtype=torch.long),
    )


def split_dataset(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    train_fraction: float,
    split_seed: int,
) -> tuple[TensorDataset, TensorDataset]:
    generator = torch.Generator().manual_seed(split_seed)
    permutation = torch.randperm(inputs.size(0), generator=generator)
    split_index = int(train_fraction * inputs.size(0))

    train_indices = permutation[:split_index]
    val_indices = permutation[split_index:]

    train_dataset = TensorDataset(inputs[train_indices], targets[train_indices])
    val_dataset = TensorDataset(inputs[val_indices], targets[val_indices])
    return train_dataset, val_dataset


def make_datasets_and_loader(
    config: ExperimentConfig,
) -> tuple[TensorDataset, TensorDataset, DataLoader]:
    inputs, targets = build_modular_dataset(config)
    train_dataset, val_dataset = split_dataset(
        inputs=inputs,
        targets=targets,
        train_fraction=config.train_fraction,
        split_seed=config.split_seed,
    )

    batch_size = min(config.batch_size, len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    return train_dataset, val_dataset, train_loader


def decode_example(tokens: torch.Tensor, target: torch.Tensor, config: ExperimentConfig) -> str:
    x, _, y, _ = tokens.tolist()
    return f"{x} {OP_SYMBOLS[config.operation]} {y} = {int(target)}"


class GrokkingTransformer(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.sequence_length = 4
        vocab_size = config.prime + 2

        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(self.sequence_length, config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.mlp_mult * config.d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.prime, bias=False)

        causal_mask = torch.triu(
            torch.full((self.sequence_length, self.sequence_length), float("-inf")),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.size(1), device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions).unsqueeze(0)
        hidden = self.transformer(
            hidden,
            mask=self.causal_mask[: tokens.size(1), : tokens.size(1)],
        )
        hidden = self.final_norm(hidden[:, -1, :])
        return self.output(hidden)


class PaperTransformer(nn.Module):
    """The post-norm decoder architecture in the authors' released code."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(PAPER_VOCAB_SIZE, config.d_model)
        positions = torch.arange(6, dtype=torch.float32).unsqueeze(1)
        dimensions = torch.arange(0, config.d_model, 2, dtype=torch.float32)
        angles = positions / (10000 ** (dimensions / config.d_model))
        position_encoding = torch.empty(6, config.d_model)
        position_encoding[:, 0::2] = torch.sin(angles)
        position_encoding[:, 1::2] = torch.cos(angles)
        self.register_buffer("position_encoding", position_encoding, persistent=False)
        self.blocks = nn.ModuleList(
            [PaperDecoderBlock(config.d_model, config.n_heads, config.mlp_mult) for _ in range(config.n_layers)]
        )
        self.output = nn.Linear(config.d_model, PAPER_VOCAB_SIZE, bias=False)
        self.register_buffer("causal_mask", torch.ones(6, 6, dtype=torch.bool).tril(), persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(tokens) + self.position_encoding[: tokens.size(1)]
        mask = self.causal_mask[: tokens.size(1), : tokens.size(1)]
        for block in self.blocks:
            hidden = block(hidden, mask)
        # Predict answer after '=' and EOS after the answer, matching y_rhs.
        return self.output(hidden[:, -2:, :])


class PaperDecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_mult: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = nn.ModuleList([nn.Linear(d_model, self.d_head, bias=False) for _ in range(n_heads)])
        self.k = nn.ModuleList([nn.Linear(d_model, self.d_head, bias=False) for _ in range(n_heads)])
        self.v = nn.ModuleList([nn.Linear(d_model, self.d_head, bias=False) for _ in range(n_heads)])
        self.attn_out = nn.Linear(d_model, d_model, bias=False)
        self.attn_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_mult * d_model, bias=False),
            nn.ReLU(),
            nn.Linear(mlp_mult * d_model, d_model, bias=False),
        )
        self.mlp_norm = nn.LayerNorm(d_model)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        heads = []
        for q_proj, k_proj, v_proj in zip(self.q, self.k, self.v):
            q, k, v = q_proj(hidden), k_proj(hidden), v_proj(hidden)
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
            scores = scores.masked_fill(~mask, float("-inf"))
            heads.append(torch.softmax(scores, dim=-1) @ v)
        hidden = self.attn_norm(hidden + self.attn_out(torch.cat(heads, dim=-1)))
        return self.mlp_norm(hidden + self.mlp(hidden))


@torch.no_grad()
def evaluate_model(model: nn.Module, dataset: TensorDataset) -> dict[str, float]:
    model.eval()
    batch_size = min(2048, len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for tokens, targets in loader:
        tokens = tokens.to(DEVICE)
        targets = targets.to(DEVICE)

        logits = model(tokens)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        total_loss += float(loss.item()) * targets.size(0)
        predictions = logits.argmax(dim=-1)
        if targets.ndim == 2:
            # Reference metric: an equation is correct only when every RHS
            # token (the answer and trailing EOS) is correct.
            total_correct += int((predictions == targets).all(dim=-1).sum().item())
        else:
            total_correct += int((predictions == targets).sum().item())
        total_examples += targets.size(0)

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def train_experiment(
    config: ExperimentConfig,
    *,
    paper_figure1: bool = False,
) -> tuple[list[dict[str, float]], nn.Module]:
    seed_everything(config.model_seed)

    if paper_figure1:
        inputs, targets = build_figure1_dataset(config)
        split_index = round(config.train_fraction * len(inputs))
        train_dataset = TensorDataset(inputs[:split_index], targets[:split_index])
        val_dataset = TensorDataset(inputs[split_index:], targets[split_index:])
        batch_size = min(config.batch_size, math.ceil(len(train_dataset) / 2))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        model: nn.Module = PaperTransformer(config).to(DEVICE)
    else:
        train_dataset, val_dataset, train_loader = make_datasets_and_loader(config)
        model = GrokkingTransformer(config).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        # The released implementation uses base lr=1 and has LambdaLR return
        # the absolute learning rate. Preserve that detail for Figure 1.
        lr=1.0 if paper_figure1 else config.lr,
        betas=(0.9, 0.98),
        weight_decay=config.weight_decay,
    )

    def lr_multiplier(step: int) -> float:
        if config.warmup_steps <= 0:
            return config.lr if paper_figure1 else 1.0
        if paper_figure1:
            return min(step / config.warmup_steps, 1.0) * config.lr
        return min((step + 1) / config.warmup_steps, 1.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
    loss_fn = nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
    loader_iter = iter(train_loader)

    for step in range(1, config.max_steps + 1):
        try:
            tokens, targets = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            tokens, targets = next(loader_iter)

        model.train()
        tokens = tokens.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step == 1 or step % config.eval_every == 0 or step == config.max_steps:
            train_metrics = evaluate_model(model, train_dataset)
            val_metrics = evaluate_model(model, val_dataset)
            history.append(
                {
                    "step": float(step),
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                }
            )

    return history, model


def plot_figure1_left(histories: list[list[dict[str, float]]]) -> Figure:
    fig, axis = plt.subplots(figsize=(6.4, 4.1))
    max_step = 1
    for seed_index, history in enumerate(histories):
        steps = [int(row["step"]) for row in history]
        max_step = max(max_step, max(steps))
        axis.plot(
            steps,
            [100 * row["train_accuracy"] for row in history],
            color="#ff0000",
            alpha=0.7,
            linewidth=1,
            label="train" if seed_index == 0 else None,
        )
        axis.plot(
            steps,
            [100 * row["val_accuracy"] for row in history],
            color="#008000",
            alpha=0.7,
            linewidth=1,
            label="val" if seed_index == 0 else None,
        )
    axis.set_xscale("log")
    axis.set_xlim(5, max_step)
    axis.set_ylim(-2, 102)
    axis.set_title("Modular Division (training on 50% of data)", fontsize=10)
    axis.set_xlabel("Optimization Steps")
    axis.set_ylabel("Accuracy")
    axis.legend()
    fig.tight_layout()
    return fig


def first_step_at_accuracy(
    history: list[dict[str, float]],
    threshold: float = 0.99,
    split: str = "val",
) -> int | None:
    key = f"{split}_accuracy"
    for row in history:
        if row[key] >= threshold:
            return int(row["step"])
    return None


def plot_training_curves(history: list[dict[str, float]], title: str) -> Figure:
    steps = [int(row["step"]) for row in history]
    train_acc = [row["train_accuracy"] for row in history]
    val_acc = [row["val_accuracy"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(steps, train_acc, label="train", linewidth=2)
    axes[0].plot(steps, val_acc, label="validation", linewidth=2)
    axes[0].set_xscale("log")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_title(f"{title}: accuracy")
    axes[0].set_xlabel("optimization step")
    axes[0].set_ylabel("accuracy")
    axes[0].legend()

    axes[1].plot(steps, train_loss, label="train", linewidth=2)
    axes[1].plot(steps, val_loss, label="validation", linewidth=2)
    axes[1].set_xscale("log")
    axes[1].set_title(f"{title}: loss")
    axes[1].set_xlabel("optimization step")
    axes[1].set_ylabel("cross-entropy")
    axes[1].legend()

    plt.tight_layout()
    return fig


def summarize_run(history: list[dict[str, float]], label: str) -> None:
    last = history[-1]
    print(label)
    print(f"  final train accuracy: {last['train_accuracy']:.3f}")
    print(f"  final val accuracy:   {last['val_accuracy']:.3f}")
    print(f"  first train step >= 99%: {first_step_at_accuracy(history, split='train')}")
    print(f"  first val step >= 99%:   {first_step_at_accuracy(history, split='val')}")


def plot_weight_decay_summary(summary: list[dict[str, float]], title: str) -> Figure:
    weight_decays = [row["weight_decay"] for row in summary]
    final_accuracies = [row["final_val_accuracy"] for row in summary]
    steps_to_99 = [row["first_val_99_step"] for row in summary]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(weight_decays, final_accuracies, marker="o", linewidth=2)
    axes[0].set_xscale("symlog", linthresh=1e-4)
    axes[0].set_title(f"{title}: final validation accuracy")
    axes[0].set_xlabel("weight decay")
    axes[0].set_ylabel("final validation accuracy")

    axes[1].plot(weight_decays, steps_to_99, marker="o", linewidth=2)
    axes[1].set_xscale("symlog", linthresh=1e-4)
    axes[1].set_yscale("log")
    axes[1].set_title(f"{title}: step to 99% validation accuracy")
    axes[1].set_xlabel("weight decay")
    axes[1].set_ylabel("optimization step")

    plt.tight_layout()
    return fig


def plot_train_fraction_sweep(summary: list[dict[str, float]], title: str) -> Figure:
    fractions = [row["train_fraction"] for row in summary]
    final_accuracies = [row["final_val_accuracy"] for row in summary]
    steps_to_99 = [row["first_val_99_step"] for row in summary]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(fractions, final_accuracies, marker="o", linewidth=2)
    axes[0].set_title(f"{title}: final validation accuracy")
    axes[0].set_xlabel("train fraction")
    axes[0].set_ylabel("final validation accuracy")

    axes[1].plot(fractions, steps_to_99, marker="o", linewidth=2)
    axes[1].set_yscale("log")
    axes[1].set_title(f"{title}: step to 99% validation accuracy")
    axes[1].set_xlabel("train fraction")
    axes[1].set_ylabel("optimization step")

    plt.tight_layout()
    return fig


def run_weight_decay_ablation(
    base_config: ExperimentConfig,
    weight_decays: list[float],
) -> tuple[dict[float, list[dict[str, float]]], list[dict[str, float]]]:
    histories: dict[float, list[dict[str, float]]] = {}
    summary: list[dict[str, float]] = []

    for weight_decay in weight_decays:
        config = replace(base_config, weight_decay=weight_decay)
        history, _ = train_experiment(config)
        histories[weight_decay] = history
        last = history[-1]
        summary.append(
            {
                "weight_decay": weight_decay,
                "final_val_accuracy": last["val_accuracy"],
                "first_val_99_step": float(first_step_at_accuracy(history, split="val") or math.nan),
            }
        )

    return histories, summary


def run_train_fraction_sweep(
    base_config: ExperimentConfig,
    train_fractions: list[float],
) -> list[dict[str, float]]:
    summary: list[dict[str, float]] = []

    for train_fraction in train_fractions:
        config = replace(base_config, train_fraction=train_fraction)
        history, _ = train_experiment(config)
        last = history[-1]
        summary.append(
            {
                "train_fraction": train_fraction,
                "final_val_accuracy": last["val_accuracy"],
                "first_val_99_step": float(first_step_at_accuracy(history, split="val") or math.nan),
            }
        )

    return summary


def save_or_show_figure(
    fig: Figure,
    name: str,
    output_dir: Path | None,
    show_plots: bool,
) -> None:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def print_preview_example() -> None:
    preview_config = replace(BASE_CONFIG, prime=13, operation="div")
    preview_inputs, preview_targets = build_modular_dataset(preview_config)
    print("Preview examples:")
    for idx in range(5):
        print(f"  {decode_example(preview_inputs[idx], preview_targets[idx], preview_config)}")


def run_smoke(output_dir: Path | None, show_plots: bool) -> None:
    smoke_config = replace(
        BASE_CONFIG,
        prime=11,
        operation="add",
        train_fraction=0.9,
        d_model=64,
        batch_size=128,
        weight_decay=1.0,
        max_steps=1500,
        eval_every=25,
    )
    history, _ = train_experiment(smoke_config)
    summarize_run(history, "Smoke run: modular addition")
    fig = plot_training_curves(history, "Smoke run")
    save_or_show_figure(fig, "smoke_run", output_dir, show_plots)


def run_delayed_generalization(output_dir: Path | None, show_plots: bool, steps: int | None) -> None:
    config = replace(
        BASE_CONFIG,
        prime=97,
        operation="div",
        train_fraction=0.5,
        weight_decay=0.0,
        max_steps=steps or 1_000_000,
        eval_every=1_000,
    )
    histories = []
    for model_seed in range(3):
        seeded_config = replace(config, model_seed=model_seed)
        history, _ = train_experiment(seeded_config, paper_figure1=True)
        histories.append(history)
        summarize_run(history, f"Figure 1 left: seed {model_seed}")
    fig = plot_figure1_left(histories)
    save_or_show_figure(fig, "figure_1_left", output_dir, show_plots)


def run_weight_decay_mode(output_dir: Path | None, show_plots: bool, steps: int | None) -> None:
    base_config = replace(
        BASE_CONFIG,
        prime=97,
        operation="div",
        train_fraction=0.5,
        max_steps=steps or 10_000,
        eval_every=250,
    )
    _, summary = run_weight_decay_ablation(
        base_config,
        weight_decays=[0.0, 1e-4, 1e-2, 0.1, 1.0],
    )
    for row in summary:
        print(row)
    fig = plot_weight_decay_summary(summary, "Weight decay ablation")
    save_or_show_figure(fig, "weight_decay_ablation", output_dir, show_plots)


def run_fraction_sweep_mode(output_dir: Path | None, show_plots: bool, steps: int | None) -> None:
    base_config = replace(
        BASE_CONFIG,
        prime=97,
        operation="div",
        weight_decay=1.0,
        max_steps=steps or 12_000,
        eval_every=250,
    )
    summary = run_train_fraction_sweep(
        base_config,
        train_fractions=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    )
    for row in summary:
        print(row)
    fig = plot_train_fraction_sweep(summary, "Train-fraction sweep")
    save_or_show_figure(fig, "train_fraction_sweep", output_dir, show_plots)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="First-pass grokking reproduction script converted from the notebook.",
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "delayed", "weight-decay", "fraction-sweep"],
        default="smoke",
        help="Which experiment to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where plots should be saved. If omitted, plots are only shown unless --no-show is set.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override the default number of optimization steps for the selected mode.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open plot windows. Useful for non-interactive runs.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    seed_everything(42)
    print(f"Using device: {DEVICE}")
    print_preview_example()

    show_plots = not args.no_show

    if args.mode == "smoke":
        run_smoke(args.output_dir, show_plots)
    elif args.mode == "delayed":
        run_delayed_generalization(args.output_dir, show_plots, args.steps)
    elif args.mode == "weight-decay":
        run_weight_decay_mode(args.output_dir, show_plots, args.steps)
    elif args.mode == "fraction-sweep":
        run_fraction_sweep_mode(args.output_dir, show_plots, args.steps)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
