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

## License

MIT
