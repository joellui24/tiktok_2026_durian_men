# Optional FM training source

These files are retained for model transparency and retraining. They are not imported by the private evaluator and are not required for inference.

Install the additional training dependency from the repository root:

```bash
python -m pip install -r requirements-training.txt
```

The trainer requires the original frozen 50,000-product `catalog.jsonl`, which is intentionally not duplicated in the submission. Run a small reproducibility check with an organizer-provided catalog:

```bash
python training/train_fm.py \
  --catalog /path/to/catalog.jsonl \
  --trajectory-count 25000 \
  --variant hybrid \
  --output training/outputs/fm_candidate.sqlite3 \
  --metrics training/outputs/training_metrics.json \
  --manifest training/outputs/dataset_manifest.json \
  --negative-audit training/outputs/negative_audit.csv \
  --cross-audit training/outputs/cross_weights.csv
```

`fm_training.py` contains the trainer, `trajectory_data.py` constructs catalog-only synthetic trajectories, and `train_fm.py` is the command-line entry point. The selected experiment configuration is preserved in `configs/fm_trajectory_v2.json`.
