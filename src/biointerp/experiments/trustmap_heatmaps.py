"""TrustMap heatmaps on ESM-2 (PLM-native).

Per-residue reliability map: for example proteins, mask every position in turn, compute
the probe score (trained on the extraction npz) and correctness; save JSON + an HTML heatmap.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from biointerp.core import (
    DEFAULT_NPZ,
    RESULTS_DIR,
    load_sequences,
    split_by_protein,
    univariate_auroc,
)
from biointerp.esm import (
    AA_ALPHABET,
    aa_token_ids,
    get_ffn_output_proj,
    get_transformer_layers,
    load_model,
    precompute_col_norms,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", default=str(DEFAULT_NPZ))
    p.add_argument("--fasta", default=None)
    p.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--protein-ids", default="0,5")
    p.add_argument("--max-len", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    return p.parse_args()


@torch.inference_mode()
def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from biointerp.core import DEFAULT_FASTA

    args = parse_args()
    data = np.load(args.npz)
    y = (~data["correct"]).astype(int)
    protein = data["protein"]
    int(data["intermediate_dim"])
    tr, _ = split_by_protein(protein, seed=args.seed)
    au = univariate_auroc(data["cett"][tr], y[tr])
    idx = np.argsort(au)[::-1][: args.top_k]
    scaler = StandardScaler()
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=args.seed)
    clf.fit(scaler.fit_transform(data["cett"][tr][:, idx]), y[tr])
    print(f"probe: top-{args.top_k} features, train n={int(tr.sum())}")

    seqs = load_sequences(args.fasta or str(DEFAULT_FASTA))
    model, tokenizer, device = load_model(args.model, args.device)
    blocks = get_transformer_layers(model)
    n_layers = len(blocks)
    col_norms = precompute_col_norms(model, list(range(n_layers)))
    aa_ids = aa_token_ids(tokenizer)
    aa_t = torch.tensor(aa_ids, device=device)

    z_cache = {}
    h_cache = {}

    def make_hook(layer):
        def hook(module, inputs, output):
            z_cache[layer] = inputs[0][0, mask_pos, :].float().cpu()
            h_cache[layer] = output[0, mask_pos, :].float().cpu()
            return output

        return hook

    handles = [
        get_ffn_output_proj(blocks[l]).register_forward_hook(make_hook(l)) for l in range(n_layers)
    ]
    results = {}
    t0 = time.time()
    try:
        for pid in [int(x) for x in args.protein_ids.split(",")]:
            seq = seqs[pid][: args.max_len]
            L = len(seq)
            rows = []
            for p in range(L):
                ids = list(tokenizer(seq, add_special_tokens=True)["input_ids"])
                mask_pos = p + 1
                ids[mask_pos] = tokenizer.mask_token_id
                tokens = {"input_ids": torch.tensor([ids], device=device)}
                out = model(**tokens)
                logits = out.logits[0, mask_pos]
                pred = int(logits[aa_t].argmax().item())
                cett = []
                for layer in range(n_layers):
                    z = z_cache[layer]
                    h = h_cache[layer]
                    h_norm = float(h.norm()) + 1e-8
                    cett.append(torch.abs(z) * col_norms[layer] / h_norm)
                cett = torch.cat(cett).numpy()
                score = float(clf.predict_proba(scaler.transform(cett[idx].reshape(1, -1)))[0, 1])
                rows.append(
                    {
                        "pos": p,
                        "aa": seq[p],
                        "pred": AA_ALPHABET[pred],
                        "correct": bool(pred == AA_ALPHABET.index(seq[p])),
                        "score": round(score, 4),
                    }
                )
            results[pid] = {"protein_id": pid, "length": L, "positions": rows}
            acc = np.mean([r["correct"] for r in rows])
            print(f"protein {pid}: len {L} | acc {acc:.3f} | {time.time() - t0:.0f}s")
    finally:
        for handle in handles:
            handle.remove()

    out_json = RESULTS_DIR / "trustmap_heatmaps.json"
    out_json.write_text(json.dumps(results, indent=2) + "\n")

    cells = []
    for pid, res in results.items():
        row = []
        for r in res["positions"]:
            s = r["score"]
            rr = int(255 * s)
            gg = int(200 * (1 - s))
            border = "border:2px solid black;" if not r["correct"] else ""
            row.append(
                f'<td style="background:rgb({rr},{gg},60);{border}text-align:center;padding:2px;">'
                f"{html.escape(r['aa'])}<br><small>{s:.2f}</small></td>"
            )
        cells.append(
            f'<tr><td style="white-space:nowrap;">protein {pid} (len {res["length"]})</td>{"".join(row)}</tr>'
        )
    out_html = RESULTS_DIR / "trustmap_heatmaps.html"
    out_html.write_text(
        "<html><body><h2>TrustMap - per-residue error score (red = high)</h2>"
        "<p>Cell: residue letter + probe score. Black border = model prediction wrong at that position.</p>"
        "<table>" + "".join(cells) + "</table></body></html>\n"
    )
    print(f"\nsaved: {out_json}")
    print(f"saved: {out_html}")


if __name__ == "__main__":
    main()
