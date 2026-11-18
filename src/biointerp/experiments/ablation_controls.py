"""Ablation negative controls for error neurons.

Compares top-k error neurons with size-matched random sets and bottom-k neurons
under an equal intervention-norm budget. The small default run is intended for
local validation; the full 100-draw matrix is a later compute job.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from biointerp.ablation import (
    equalize_norms,
    global_norm,
    group_by_layer,
    random_sets,
    select_top_bottom,
    summarize_ablation,
)
from biointerp.core import DEFAULT_FASTA, DEFAULT_NPZ, RESULTS_DIR, load_sequences
from biointerp.esm import (
    AA_ALPHABET,
    aa_token_ids,
    get_ffn_output_proj,
    get_transformer_layers,
)
from biointerp.models import get_spec, load_model
from biointerp.repro import metadata
from biointerp.task import make_masked_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", default=str(DEFAULT_NPZ))
    p.add_argument("--fasta", default=str(DEFAULT_FASTA))
    p.add_argument("--assignments", default=str(RESULTS_DIR / "split_assignments.tsv"))
    p.add_argument("--model-key", default=None)
    p.add_argument("--k", type=int, default=25)
    p.add_argument("--draws", type=int, default=5)
    p.add_argument("--calib-n", type=int, default=100)
    p.add_argument("--eval-n", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=str(RESULTS_DIR / "ablation_controls.json"))
    p.add_argument("--figure", default=str(RESULTS_DIR / "ablation_controls.png"))
    p.add_argument(
        "--summarize-only",
        action="store_true",
        help="recompute the summary and figure from an existing result JSON",
    )
    return p.parse_args()


def load_split(npz: str, assignments: str):
    data = np.load(npz, allow_pickle=True)
    cett = data["cett"]
    y = (~data["correct"]).astype(int)
    protein = data["protein"]

    split_of: dict[str, str] = {}
    for line in Path(assignments).read_text().splitlines()[1:]:
        if not line.strip():
            continue
        record_id, _cluster, split = line.split("\t")
        split_of[record_id] = split

    ids = np.asarray([f"seq{int(index)}" for index in protein])
    keep = np.asarray([split_of.get(record_id) in {"train", "test"} for record_id in ids])
    is_test = np.asarray([split_of.get(record_id) == "test" for record_id in ids])
    return cett[keep], y[keep], protein[keep], ~is_test[keep], is_test[keep]


def mean_activations(
    model,
    tokenizer,
    sequences,
    layers: list[int],
    intermediate: int,
    device,
    batch_size: int,
    seed: int,
):
    blocks = get_transformer_layers(model)
    sums = {layer: torch.zeros(intermediate) for layer in layers}
    count = 0
    started = time.time()

    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        tokens, positions, _ = make_masked_batch(
            batch, tokenizer, masks_per_seq=1, seed=seed + start
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        positions = positions.to(device)
        b_idx = torch.arange(positions.shape[0], device=device).unsqueeze(1)

        def hook_factory(layer):
            def hook(module, inputs, output):
                sums[layer] += inputs[0][b_idx, positions].squeeze(1).sum(dim=0).float().cpu()
                return output

            return hook

        handles = [
            get_ffn_output_proj(blocks[layer]).register_forward_hook(hook_factory(layer))
            for layer in layers
        ]
        try:
            with torch.no_grad():
                model(**tokens)
        finally:
            for handle in handles:
                handle.remove()
        count += len(batch)
        done = min(start + batch_size, len(sequences))
        if done % (batch_size * 5) < batch_size or done == len(sequences):
            print(f"  calibration {done}/{len(sequences)} | {time.time() - started:.0f}s")

    return {layer: (value / max(count, 1)).numpy() for layer, value in sums.items()}


def run_condition(
    model,
    tokenizer,
    sequences,
    layers: list[int],
    aa_t: torch.Tensor,
    aa_index: dict[str, int],
    device,
    batch_size: int,
    seed: int,
    intervention: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
):
    blocks = get_transformer_layers(model)
    correct: list[bool] = []
    probabilities_true: list[float] = []
    handles = []

    if intervention:
        for layer, (local, mean_vector) in intervention.items():
            local_t = torch.tensor(local, dtype=torch.long)
            mean_t = torch.tensor(mean_vector, dtype=torch.float32)

            def make_hook(local_idx, mean):
                def hook(module, inputs, output):
                    z = inputs[0]
                    idx = local_idx.to(z.device)
                    delta = (mean.to(z.device) - z[..., idx]) @ module.weight[:, idx].T
                    return output + delta

                return hook

            handles.append(
                get_ffn_output_proj(blocks[layer]).register_forward_hook(make_hook(local_t, mean_t))
            )

    try:
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            tokens, positions, true_aa = make_masked_batch(
                batch, tokenizer, masks_per_seq=1, seed=seed + start
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            positions = positions.to(device)
            b_idx = torch.arange(positions.shape[0], device=device).unsqueeze(1)
            with torch.no_grad():
                out = model(**tokens)
            logits = out.logits[b_idx, positions][..., aa_t].squeeze(1)
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            true_flat = [aa for row in true_aa for aa in row]
            true_index = np.asarray([aa_index[aa] for aa in true_flat])
            correct.extend((probs.argmax(axis=1) == true_index).tolist())
            probabilities_true.extend(probs[np.arange(len(true_index)), true_index].tolist())
    finally:
        for handle in handles:
            handle.remove()

    return {
        "accuracy": round(float(np.mean(correct)), 4),
        "p_true": round(float(np.mean(probabilities_true)), 4),
        "n": int(len(correct)),
    }


def plot_ablation(path: str, rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib
    except ImportError:
        print("matplotlib not installed; skipping figure")
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = [str(row["condition"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, metric, label in (
        (axes[0], "accuracy", "masked accuracy"),
        (axes[1], "p_true", "p(true)"),
    ):
        values = [float(row[metric]) for row in rows]
        axis.bar(range(len(rows)), values, color="steelblue")
        baseline = next(float(row[metric]) for row in rows if row["condition"] == "baseline")
        axis.axhline(baseline, color="black", linestyle="--", linewidth=1)
        axis.set_xticks(range(len(rows)))
        axis.set_xticklabels(conditions, rotation=35, ha="right", fontsize=8)
        axis.set_ylabel(label)
        axis.set_ylim(min(values) - 0.02, max(values) + 0.02)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    started = time.time()

    if args.summarize_only:
        payload = json.loads(Path(args.out).read_text())
        target_norm = float(payload["summary"]["equal_norm_target"])
        summary = summarize_ablation(payload["conditions"], target_norm)
        payload["summary"] = summary
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
        plot_ablation(args.figure, payload["conditions"])
        print(f"summary: {summary}")
        print(f"wrote: {args.out}")
        return
    cett, y, protein, train, test = load_split(args.npz, args.assignments)
    sequences = load_sequences(args.fasta)
    if len(sequences) < len(protein):
        raise SystemExit("FASTA has fewer sequences than the feature matrix")

    train_index = np.flatnonzero(train)
    test_index = np.flatnonzero(test)
    if len(train_index) < args.calib_n or len(test_index) < args.eval_n:
        raise SystemExit("not enough rows for the requested calibration/eval sizes")
    calib_index = train_index[: args.calib_n]
    eval_index = test_index[: args.eval_n]

    from biointerp.runner import load_config

    model_key = args.model_key or str(load_config()["model"])
    spec = get_spec(model_key)
    if spec.get("adapter") != "esm":
        raise SystemExit(f"ablation adapter {spec.get('adapter')!r} not implemented")

    model, tokenizer, device = load_model(model_key, args.device)
    layers = list(range(len(get_transformer_layers(model))))
    config = getattr(model, "config", None)
    intermediate = int(getattr(config, "intermediate_size", None) or model.intermediate_size)
    aa_ids = aa_token_ids(tokenizer)
    aa_t = torch.tensor(aa_ids, device=device)
    aa_index = {aa: index for index, aa in enumerate(AA_ALPHABET)}

    calib_sequences = [sequences[int(index)] for index in calib_index]
    eval_sequences = [sequences[int(index)] for index in eval_index]
    mean_z = mean_activations(
        model,
        tokenizer,
        calib_sequences,
        layers,
        intermediate,
        device,
        args.batch_size,
        args.seed,
    )

    top, bottom = select_top_bottom(cett, y, train, args.k)
    random = random_sets(cett.shape[1], args.k, args.draws, args.seed)

    # The target norm is the top-k condition's raw norm; every condition is
    # rescaled to it before intervention.
    top_grouped = group_by_layer(top, intermediate)
    top_raw = {layer: mean_z[layer][np.asarray(local)] for layer, local in top_grouped.items()}
    target_norm = global_norm(top_raw)

    def build(indices):
        grouped = group_by_layer(indices, intermediate)
        raw = {layer: mean_z[layer][np.asarray(local)] for layer, local in grouped.items()}
        scaled = equalize_norms(raw, target_norm)
        return {layer: (np.asarray(grouped[layer]), scaled[layer]) for layer in grouped}

    conditions = [
        ("baseline", None),
        ("top", build(top)),
        ("bottom", build(bottom)),
    ]
    for index, indices in enumerate(random):
        conditions.append((f"random-{index}", build(indices)))

    rows = []
    for name, intervention in conditions:
        result = run_condition(
            model,
            tokenizer,
            eval_sequences,
            layers,
            aa_t,
            aa_index,
            device,
            args.batch_size,
            args.seed,
            intervention,
        )
        rows.append(
            {"condition": name, "n_neurons": 0 if intervention is None else args.k, **result}
        )
        print(f"{name:12s} acc {result['accuracy']:.4f} | p(true) {result['p_true']:.4f}")

    summary = summarize_ablation(rows, target_norm)

    result = {
        "meta": metadata(),
        "model": model_key,
        "k": args.k,
        "draws": args.draws,
        "n_calib": len(calib_sequences),
        "n_eval": len(eval_sequences),
        "conditions": rows,
        "summary": summary,
        "elapsed_s": round(time.time() - started, 1),
    }
    out = Path(args.out)
    out.write_text(json.dumps(result, indent=2) + "\n")
    plot_ablation(args.figure, rows)
    print(f"summary: {summary}")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
