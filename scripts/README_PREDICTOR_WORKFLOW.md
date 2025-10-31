# Predictor Training Workflow

Quick reference for training iteration predictors from solver logs.

## Prerequisites

```bash
pip install pandas scikit-learn xgboost lightgbm treelite tl2cgen joblib
```

## Workflow

### 1. Collect Logs

Run solver with `INFO` log level to capture feature logs:

```bash
export CUOPT_LOG_LEVEL=INFO
./solver problem.mps > logs/problem.log 2>&1
```

### 2. Parse Logs

Extract training data for specific algorithm:

```bash
# Parse Feasibility Pump logs
python scripts/determinism_logs_parse.py logs/ --algorithm FP -o fp_data.pkl

# Parse PDLP (LP solver) logs
python scripts/determinism_logs_parse.py logs/ --algorithm PDLP -o pdlp_data.pkl

# Parse Constraint Propagation logs
python scripts/determinism_logs_parse.py logs/ --algorithm CP -o cp_data.pkl

# Parse Feasibility Jump legacy logs
python scripts/determinism_logs_parse.py logs/ --algorithm FJ -o fj_data.pkl
```

**The parser shows real-time progress:**
- File scanning progress
- Grep execution on N files
- Line processing progress (updates every 10,000 lines)
- Pairing progress (updates every 10 files)
- Final statistics and file size

**IMPORTANT - Log Filtering:**
The parser uses grep with EXACT pattern matching to be highly efficient:
- Pattern: `FP_FEATURES:` and `FP_RESULT:` (with colon suffix)
- Only matches predictor log lines, ignores all other FP-related logs
- Example: A log with 100,000 lines might have 10,000 lines with "FP" but only 200 predictor lines
- Grep filters these down BEFORE Python processing, making it extremely fast even with noisy logs

### 3. Inspect Features (Optional)

See what features are available:

```bash
python scripts/train_regressor.py fp_data.pkl --regressor xgboost --list-features
```

### 4. Train Model

Train with XGBoost or LightGBM:

```bash
# XGBoost with early stopping and C++ code generation
python scripts/train_regressor.py fp_data.pkl \
    --regressor xgboost \
    --seed 42 \
    --early-stopping 20 \
    --treelite-compile 8

# LightGBM with stratified split
python scripts/train_regressor.py pdlp_data.pkl \
    --regressor lightgbm \
    --seed 42 \
    --stratify-split \
    --early-stopping 20 \
    --treelite-compile 8
```

### 5. Use Generated Code

The trained model will be exported to C++ code in `./models/<name>_c_code/`:

- `header.h` - Class declaration
- `main.cpp` - Implementation
- `quantize.cpp` - Quantization helpers

Copy to `cpp/src/utilities/models/<algorithm>_predictor/` and integrate into solver.

## Log Format Reference

### FP (Feasibility Pump)
```
FP_FEATURES: n_variables=100 n_constraints=50 n_integer_vars=80 ...
FP_RESULT: iterations=142 time_taken=5.234 termination=FEASIBLE_LP_PROJECTION
```

### PDLP (LP Solver)
```
PDLP_FEATURES: n_variables=100 n_constraints=50 nnz=450 ...
PDLP_RESULT: iterations=237 time_ms=1234 termination=2
```

### CP (Constraint Propagation)
```
CP_FEATURES: n_variables=100 n_constraints=50 n_unset_vars=25 ...
CP_RESULT: time_ms=567 termination=SUCCESS iterations=0
```

## Common Options

### Training Script

- `--regressor {xgboost,lightgbm,linear,poly2,...}` - Model type
- `--seed N` - Random seed for reproducibility
- `--test-size 0.2` - Test set proportion (default 20%)
- `--stratify-split` - Balance train/test by target distribution
- `--early-stopping N` - Early stopping patience (prevents overfitting)
- `--treelite-compile N` - Generate C++ code with N threads
- `--list-features` - Show available features and exit
- `--tune` - Use tuned hyperparameters

### Parsing Script

- `--algorithm {FP,PDLP,CP,FJ}` - Which algorithm to parse
- `-o FILE` - Output pickle file (default: `<algorithm>_data.pkl`)
- `--verbose` - Show warnings and detailed output

## Tips

1. **Collect diverse problems**: Train on variety of problem types/sizes
2. **Check train/test split**: Use `--stratify-split` if targets are imbalanced
3. **Prevent overfitting**: Use `--early-stopping` with tree models
4. **Feature selection**: Edit `FEATURES_TO_EXCLUDE` in `train_regressor.py`
5. **Reproducibility**: Always set `--seed` for consistent results

## Troubleshooting

**No entries found for algorithm X**
- Check log level is set to INFO or DEBUG
- Verify solver is executing the algorithm
- Look for `X_FEATURES` and `X_RESULT` lines in logs

**Poor model performance**
- Collect more training data
- Try different regressor types
- Use `--list-features` to identify important features
- Enable `--stratify-split` for balanced splits

**C++ code generation fails**
- Install: `pip install treelite tl2cgen`
- Only works with XGBoost and LightGBM
- Check model trained successfully first

## Example Output

```bash
$ python scripts/determinism_logs_parse.py logs/ --algorithm FP -o fp_data.pkl

Scanning logs/ for .log files...
Found 42 log files

Parsing FP (Feasibility Pump) logs...
  Running grep on 42 files...
  Processing 3046 matching lines...
  Processed 3046 lines from 42 files
  Pairing features with results...
  Found 1523 complete entries from 42 files

  Total entries: 1523
  Unique files: 42
  Avg entries per file: 36.26
  Iterations (target): min=1, max=847, avg=142.35

Saving 1523 entries to fp_data.pkl...

======================================================================
✓ Success! Saved 1523 entries to fp_data.pkl
  File size: 2.34 MB
======================================================================

$ python scripts/train_regressor.py fp_data.pkl --regressor xgboost --seed 42

Loading data from: fp_data.pkl
Loaded 1523 entries with 29 columns

Data Split:
  Total entries: 1523
  Train entries: 1218 (34 files)
  Test entries: 305 (8 files)

Training xgboost regressor...
  Training complete!

Test Set Metrics:
  MSE:  1234.56
  RMSE: 35.14
  MAE:  22.67
  R²:   0.8542

Feature Importance:
    1. n_variables                              : 0.245123
    2. n_constraints                            : 0.187456
    3. initial_ratio_of_integers                : 0.156234
    ...

C source code generated to: ./models/fp_data_c_code/
  Contains optimized model source code (branch-annotated, quantized)

Success! Saved 1523 entries to fp_data.pkl
```
