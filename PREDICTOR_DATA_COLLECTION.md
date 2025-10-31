# Predictor Data Collection: Feature Logging

This document describes the instrumentation added to collect training data for work unit predictors.

## Overview

Three algorithms have been instrumented to log features before execution and performance metrics after completion:

1. **Feasibility Pump (FP)** - Main heuristic for finding feasible solutions
2. **PDLP** - First-order LP solver used for polytope projection
3. **Constraint Propagation (CP)** - Variable rounding with bounds propagation

Note: **Feasibility Jump (FJ)** already has a working predictor and doesn't need additional instrumentation.

## Log Format

All logs use `CUOPT_LOG_INFO` level with structured prefixes for easy parsing:

### Feasibility Pump (FP)

**Features logged before execution:**
```
FP_FEATURES: n_variables=%d n_constraints=%d n_integer_vars=%d n_binary_vars=%d
FP_FEATURES: nnz=%lu sparsity=%.6f nnz_stddev=%.6f unbalancedness=%.6f
FP_FEATURES: initial_feasibility=%d initial_excess=%.6f initial_objective=%.6f
FP_FEATURES: initial_ratio_of_integers=%.6f initial_n_integers=%d
FP_FEATURES: alpha=%.6f check_distance_cycle=%d cycle_detection_length=%d
FP_FEATURES: has_cutting_plane=%d time_budget=%.6f
```

**Results logged after execution:**
```
FP_RESULT: iterations=%d time_taken=%.6f termination=<REASON>
```

**Termination reasons:**
- `TIME_LIMIT` - Time budget exhausted
- `TIME_LIMIT_AFTER_ROUND` - Time limit during rounding phase
- `FEASIBLE_LP_PROJECTION` - Found feasible via LP projection
- `FEASIBLE_LP_VERIFIED` - Found feasible via high-precision LP
- `FEASIBLE_AFTER_ROUND` - Found feasible after rounding
- `FEASIBLE_DISTANCE_CYCLE` - Found feasible during distance cycle handling
- `INFEASIBLE_DISTANCE_CYCLE` - Distance cycle detected, no feasible found
- `ASSIGNMENT_CYCLE` - Assignment cycle detected

**Location:** `cpp/src/mip/local_search/feasibility_pump/feasibility_pump.cu::run_single_fp_descent`

---

### PDLP (LP Solver)

**Features logged before execution:**
```
PDLP_FEATURES: n_variables=%d n_constraints=%d nnz=%lu
PDLP_FEATURES: sparsity=%.6f nnz_stddev=%.6f unbalancedness=%.6f
PDLP_FEATURES: has_warm_start=%d time_limit=%.6f iteration_limit=%d
PDLP_FEATURES: tolerance=%.10f check_infeasibility=%d return_first_feasible=%d
```

**Results logged after execution:**
```
PDLP_RESULT: iterations=%d time_ms=%lld termination=%d
PDLP_RESULT: primal_objective=%.10f dual_objective=%.10f gap=%.10f
PDLP_RESULT: l2_primal_residual=%.10f l2_dual_residual=%.10f
```

**Termination status codes:**
- `0` - NoTermination
- `1` - NumericalError
- `2` - Optimal
- `3` - PrimalInfeasible
- `4` - DualInfeasible
- `5` - IterationLimit
- `6` - TimeLimit
- `7` - PrimalFeasible
- `8` - ConcurrentLimit

**Location:** `cpp/src/mip/relaxed_lp/relaxed_lp.cu::get_relaxed_lp_solution`

---

### Constraint Propagation (CP)

**Features logged before execution:**
```
CP_FEATURES: n_variables=%d n_constraints=%d n_integer_vars=%d
CP_FEATURES: nnz=%lu sparsity=%.6f
CP_FEATURES: n_unset_vars=%d initial_excess=%.6f time_budget=%.6f
CP_FEATURES: round_all_vars=%d lp_run_time_after_feasible=%.6f
```

**Results logged after execution:**
```
CP_RESULT: time_ms=%lld termination=<STATUS> iterations=%d
```

**Termination status:**
- `BRUTE_FORCE_SUCCESS` - Succeeded via simple rounding
- `SUCCESS` - Found feasible solution
- `FAILED` - Did not find feasible solution

**Location:** `cpp/src/mip/local_search/rounding/constraint_prop.cu::apply_round`

---

## Data Collection Workflow

### 1. Run Solver with Logging Enabled

Ensure the log level is set to `INFO` or higher to capture the feature logs:

```bash
export CUOPT_LOG_LEVEL=INFO
# or
export CUOPT_LOG_LEVEL=DEBUG
```

### 2. Parse Logs

Use the provided `determinism_logs_parse.py` script to automatically extract training data:

```bash
# Parse FP (Feasibility Pump) logs
python scripts/determinism_logs_parse.py logs/ --algorithm FP -o fp_data.pkl

# Parse PDLP (LP Solver) logs
python scripts/determinism_logs_parse.py logs/ --algorithm PDLP -o pdlp_data.pkl

# Parse CP (Constraint Propagation) logs
python scripts/determinism_logs_parse.py logs/ --algorithm CP -o cp_data.pkl

# Parse FJ (Feasibility Jump) legacy logs
python scripts/determinism_logs_parse.py logs/ --algorithm FJ -o fj_data.pkl
```

The script will:
- Find all `.log` files in the specified directory
- Extract all `<ALGORITHM>_FEATURES` and `<ALGORITHM>_RESULT` log lines using grep
- Pair features with results in order using line numbers
- Export to pickle format compatible with `train_regressor.py`

**Performance optimizations for large logs:**
- **Exact pattern matching**: Grep uses `FP_FEATURES:` / `FP_RESULT:` (with colon) to match ONLY predictor lines
  - Example: Log with 100K lines and 10K "FP" references → grep extracts only ~200 predictor lines
  - Filters before Python processing, so noisy logs don't slow down parsing
- Single grep call per algorithm (combines features + results)
- Uses grep's `-n` flag for line-number-based pairing
- Minimal Python string processing (simple split operations)
- Single-pass parsing with efficient dictionary accumulation
- Handles millions of log lines efficiently

**Script output (with progress indicators):**
```
Scanning logs/ for .log files...
Found 42 log files

Parsing FP (Feasibility Pump) logs...
  Running grep on 42 files...
  Processing 3046 matching lines...
  Progress: 10000/3046 lines, 42 files
  Progress: 20000/3046 lines, 42 files
  Processed 3046 lines from 42 files
  Pairing features with results...
  Pairing: 10/42 files, 362 entries found
  Pairing: 20/42 files, 724 entries found
  Pairing: 30/42 files, 1086 entries found
  Pairing: 40/42 files, 1448 entries found
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
```

**Progress updates:**
- Line processing: Every 10,000 lines
- Pairing: Every 10 files
- Uses carriage return (`\r`) for in-place updates

### 3. Train Predictor Model

Use the `train_regressor.py` script with the parsed data:

```bash
# Train XGBoost model for FP
python scripts/train_regressor.py fp_data.pkl --regressor xgboost --seed 42

# Train LightGBM model for PDLP
python scripts/train_regressor.py pdlp_data.pkl --regressor lightgbm --seed 42

# View available features before training
python scripts/train_regressor.py cp_data.pkl --regressor xgboost --list-features
```

The training script will:
- Load the pickle file
- Split data by files (train/test)
- Train the specified model
- Evaluate performance (R², RMSE, MAE)
- Export to C++ code using TL2cgen (for XGBoost/LightGBM)
- Save model and metadata

---

## Feature Descriptions

### Problem Structure Features

- **n_variables** - Total number of decision variables
- **n_constraints** - Total number of constraints
- **n_integer_vars** - Number of integer/binary variables
- **n_binary_vars** - Number of binary (0/1) variables
- **nnz** - Non-zero coefficients in constraint matrix
- **sparsity** - Matrix sparsity: nnz / (n_constraints × n_variables)
- **nnz_stddev** - Standard deviation of non-zeros per constraint row
- **unbalancedness** - Load balancing metric for constraint matrix

### Solution State Features (FP)

- **initial_feasibility** - Whether starting solution is feasible (0/1)
- **initial_excess** - Sum of constraint violations
- **initial_objective** - Objective value of initial solution
- **initial_ratio_of_integers** - Fraction of integer vars already integral
- **initial_n_integers** - Count of integer vars at integral values

### Algorithm Configuration Features (FP)

- **alpha** - Weight between original objective and distance objective
- **check_distance_cycle** - Whether distance-based cycle detection is enabled
- **cycle_detection_length** - Number of recent solutions tracked
- **has_cutting_plane** - Whether objective cutting plane was added
- **time_budget** - Allocated time in seconds

### Solver Configuration Features (PDLP)

- **has_warm_start** - Whether initial primal/dual solution provided
- **time_limit** - Time budget in seconds
- **iteration_limit** - Maximum iterations allowed
- **tolerance** - Optimality tolerance
- **check_infeasibility** - Whether to detect infeasibility
- **return_first_feasible** - Whether to return on first primal feasible

### Rounding Configuration Features (CP)

- **n_unset_vars** - Integer variables not yet set
- **round_all_vars** - Whether to round all variables or selective
- **lp_run_time_after_feasible** - Time budget for post-feasibility LP

---

## Integration with Existing Predictor

The FJ predictor already exists at `cpp/src/utilities/models/fj_predictor/`. The same workflow can be used:

1. Collect training data as described above
2. Train XGBoost model
3. Export to C++ using TreeLite (as done for FJ)
4. Integrate into solver with work unit → iteration conversion

Example from FJ predictor:
```cpp
// cpp/src/mip/feasibility_jump/feasibility_jump.cu:1283-1291
if (settings.work_unit_limit != std::numeric_limits<double>::infinity()) {
    std::map<std::string, float> features_map = get_feature_vector(0);
    float iter_prediction = std::max(
        (f_t)0.0,
        (f_t)ceil(context.work_unit_predictors.fj_predictor.predict_scalar(features_map))
    );
    CUOPT_LOG_DEBUG("FJ determ: Estimated number of iterations for %f WU: %f",
                    settings.work_unit_limit,
                    iter_prediction);
    settings.iteration_limit = std::min(settings.iteration_limit, (i_t)iter_prediction);
}
```

---

## Next Steps

1. **Collect Data**: Run solver on diverse problem sets with logging enabled
2. **Analyze**: Examine feature importance and correlation with execution time
3. **Train Models**: Build iteration predictors for FP, PDLP, and CP
4. **Validate**: Test predictors maintain solution quality while achieving determinism
5. **Deploy**: Integrate trained models into solver (similar to FJ predictor)
6. **Hierarchical Allocation**: Implement work unit budget allocation across nested algorithms

---

## Complete Workflow Example

Here's a complete end-to-end example:

### Step 1: Run Solver and Collect Logs

```bash
# Set log level to capture feature logs
export CUOPT_LOG_LEVEL=INFO

# Run your solver on test problems
./my_solver problem1.mps > logs/problem1.log 2>&1
./my_solver problem2.mps > logs/problem2.log 2>&1
# ... run on many problems
```

### Step 2: Parse Logs for Each Algorithm

```bash
# Parse FP logs
python scripts/determinism_logs_parse.py logs/ --algorithm FP -o fp_data.pkl

# Parse PDLP logs
python scripts/determinism_logs_parse.py logs/ --algorithm PDLP -o pdlp_data.pkl

# Parse CP logs
python scripts/determinism_logs_parse.py logs/ --algorithm CP -o cp_data.pkl
```

### Step 3: Inspect Features

```bash
# See what features are available for FP
python scripts/train_regressor.py fp_data.pkl --regressor xgboost --list-features
```

Output:
```
======================================================================
Available features in dataset (28 total):
======================================================================
    1. alpha
    2. check_distance_cycle
    3. cycle_detection_length
    4. has_cutting_plane
    5. initial_excess
    6. initial_feasibility
    7. initial_n_integers
    8. initial_objective
    9. initial_ratio_of_integers
   10. n_binary_vars
   11. n_constraints
   12. n_integer_vars
   13. n_variables
   14. nnz
   15. nnz_stddev
   16. sparsity
   17. time_budget
   18. unbalancedness
   ...
```

### Step 4: Train Models

```bash
# Train FP predictor with XGBoost
python scripts/train_regressor.py fp_data.pkl \
    --regressor xgboost \
    --seed 42 \
    --early-stopping 20 \
    --treelite-compile 8

# Train PDLP predictor with LightGBM
python scripts/train_regressor.py pdlp_data.pkl \
    --regressor lightgbm \
    --seed 42 \
    --early-stopping 20 \
    --treelite-compile 8
```

### Step 5: Review Results

The training script will output:
- Cross-validation scores
- Train/test metrics (R², RMSE, MAE)
- Feature importance ranking
- Sample predictions
- Worst predictions with feature values

Example output:
```
Training complete!

Test Set Metrics:
  MSE:  1234.5678
  RMSE: 35.14
  MAE:  22.67
  R²:   0.8542

Feature Importance:
    1. n_variables                              : 0.245123
    2. n_constraints                            : 0.187456
    3. initial_ratio_of_integers                : 0.156234
    4. sparsity                                 : 0.098765
    ...

C source code generated to: ./models/fp_data_c_code/
  Contains optimized model source code (branch-annotated, quantized)
```

### Step 6: Integrate into Solver

The generated C++ code will be in `./models/<algorithm>_data_c_code/`:
- `header.h` - Class declaration with predict functions
- `main.cpp` - Implementation
- `quantize.cpp` - Quantization helpers (if enabled)

Copy these files to `cpp/src/utilities/models/<algorithm>_predictor/` and integrate similar to the existing FJ predictor.

---

## Notes

- Line Segment Search was excluded as it can be predicted from FJ predictor (it runs FJ internally)
- CP iteration tracking needs enhancement (currently logs 0 iterations)
- Consider adding more dynamic features during execution for better predictions
- The termination reasons can help understand when algorithms succeed/fail
- Use `--stratify-split` when training to ensure balanced train/test distribution
- The `--early-stopping` parameter helps prevent overfitting on tree models
- TL2cgen compilation with `--treelite-compile` generates optimized C++ code with branch annotation and quantization enabled by default
