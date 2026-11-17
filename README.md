# biointerp

[![PyPI](https://img.shields.io/pypi/v/biointerp.svg)](https://pypi.org/project/biointerp/)
[![Tests](https://github.com/huseyincavusbi/biointerp/actions/workflows/tests.yml/badge.svg)](https://github.com/huseyincavusbi/biointerp/actions/workflows/tests.yml)

Mechanistic interpretability experiments for protein language models: CETT error-neuron extraction, probing, causal interventions, and variant-effect analysis.

> Research alpha. APIs and configs may change.

## Install

```bash
pip install biointerp
# or
uv add biointerp
```

For development:

```bash
git clone https://github.com/huseyincavusbi/biointerp.git
cd biointerp
uv sync
```

Python 3.11+. Models download from Hugging Face on first use.

## Quickstart

```bash
biointerp list models
biointerp list experiments
biointerp run probing_classifier --model esm2_8m --set n=50
```

Results go to `outputs/<experiment>/<timestamp>/`.

## Models and experiments

Models live in `conf/model/*.yaml` (ESM-2, ESMC, ProtGPT2, ProGen2). There are 34 experiment modules under `src/biointerp/experiments/`; configured runs live in `conf/experiment/`.

Protein LMs only for now; DNA support is in progress.

## Carbon-X environment

Carbon-X requires `transformers>=5` and is therefore kept out of the main lock
(`4.57.6`). Use a separate environment; optional extras cannot isolate a
conflicting Transformers major version.

```bash
uv venv .venv-carbonx --python 3.13
uv pip install --python .venv-carbonx/bin/python -r requirements/carbonx.txt
```

Prompt/token harness only, no model weights:

```bash
uv run python -m biointerp.experiments.carbonx_context \
  --ref ACGTACGTACGTACGTACGT \
  --alt ACGTACGTACGTACGTACGA \
  --dry-run
```

Model runs need a GPU with roughly 30 GB for 14.8B bf16. The 14.8B checkpoint
never enters the default unit-test matrix. See `requirements/carbonx.txt` for
the pinned setup.

## License

MIT
