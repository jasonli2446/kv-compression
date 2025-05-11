# KV Compression in Transformers

This project provides a modular framework for experimenting with KV cache compression in transformer models.

## Structure

- `models/`: Simplified transformer model implementation
- `compression/`: SVD and Tucker compression modules
- `cache/`: KV cache management and compression-aware logic
- `evaluation/`: Perplexity and memory tracking
- `scripts/`: CLI scripts to run experiments
- `configs/`: JSON/YAML config templates
- `tests/`: Unit tests for each module

## Usage

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run an experiment:
   ```bash
   python scripts/run_experiment.py --config configs/experiment.yaml
   ```
3. Run tests:
   ```bash
   pytest
   ```

## Notes

- For Tucker compression, install `tensorly`.
- The code is for research and experimentation, not production.
