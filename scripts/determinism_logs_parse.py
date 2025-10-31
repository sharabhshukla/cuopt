#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Parse log files containing algorithm feature logs and export to pickle format for training.

Supports parsing of:
- FP (Feasibility Pump): FP_FEATURES and FP_RESULT logs
- PDLP (LP Solver): PDLP_FEATURES and PDLP_RESULT logs
- CP (Constraint Propagation): CP_FEATURES and CP_RESULT logs
- FJ (Feasibility Jump): Legacy FJ: format

IMPORTANT - Grep Specificity:
The parser uses EXACT pattern matching with grep to filter logs efficiently.
For example, when parsing FP logs:
- Grep pattern: 'FP_FEATURES:' and 'FP_RESULT:' (with colon)
- Matches ONLY predictor log lines, NOT general FP debug/info lines
- A log with 10,000 lines containing "FP" might have only 100 predictor lines
- Grep filters down to only the relevant lines before Python processing

Performance optimizations for very large log files:
- Single grep call per algorithm (instead of separate calls for features/results)
- Uses grep's -n flag to get line numbers for efficient pairing
- Minimal Python string processing (split instead of regex)
- Single-pass parsing with dictionary accumulation
- Avoids redundant string operations
- DRY refactoring: Generic parser eliminates duplicate code (~105 lines removed)
- Real-time progress indicators (every 10K lines, every 10 files)

Usage:
    python determinism_logs_parse.py <input_directory> --algorithm FP [-o output.pkl]
    python determinism_logs_parse.py <input_directory> --algorithm PDLP [-o output.pkl]
    python determinism_logs_parse.py <input_directory> --algorithm CP [-o output.pkl]
    python determinism_logs_parse.py <input_directory> --algorithm FJ [-o output.pkl]
"""

import argparse
import pickle
import subprocess
import os
import glob
import re
from typing import List, Dict, Any, Optional, Tuple


SUPPORTED_ALGORITHMS = ['FP', 'PDLP', 'CP', 'FJ']


def parse_value(value_str: str) -> Any:
    """Convert string value to appropriate type (int, float, or str)."""
    try:
        # Try to parse as float if it contains a decimal point or scientific notation
        if '.' in value_str or 'e' in value_str.lower():
            return float(value_str)
        else:
            return int(value_str)
    except ValueError:
        # Keep as string if conversion fails
        return value_str


def parse_key_value_line(line: str, prefix: str) -> Dict[str, Any]:
    """
    Parse a line containing key=value pairs after removing prefix.

    Example
    -------
        "FP_FEATURES: n_variables=100 n_constraints=50"
        -> {'n_variables': 100, 'n_constraints': 50}
    """
    entry = {}

    # Remove prefix
    if prefix in line:
        line = line.split(prefix, 1)[1].strip()

    # Parse key=value pairs
    # Handle both space-separated and comma-separated
    for kv_pair in line.split():
        if '=' in kv_pair:
            key, value = kv_pair.split('=', 1)
            # Remove trailing commas
            value = value.rstrip(',')
            entry[key] = parse_value(value)

    return entry


def parse_generic_algorithm_logs(log_files: List[str],
                                 algorithm: str,
                                 algorithm_name: str) -> List[Dict[str, Any]]:
    """
    Generic parser for algorithm feature and result logs.

    Matches <ALGORITHM>_FEATURES lines with subsequent <ALGORITHM>_RESULT lines.
    Uses grep efficiently to minimize Python-side processing.

    Args:
        log_files: List of log file paths to parse
        algorithm: Algorithm prefix (e.g., 'FP', 'PDLP', 'CP')
        algorithm_name: Full name for display (e.g., 'Feasibility Pump')

    Returns
    -------
        List of dictionaries with combined features and results
    """
    print(f"\nParsing {algorithm} ({algorithm_name}) logs...")
    print(f"  Running grep on {len(log_files)} files...")

    # Construct grep patterns with EXACT match requirements
    # The colon at the end ensures we ONLY match the feature/result log lines
    # and ignore all other lines containing the algorithm name
    features_pattern = f'{algorithm}_FEATURES:'
    result_pattern = f'{algorithm}_RESULT:'

    # Use grep with:
    # -H: Always print filename (even with single file)
    # -n: Print line numbers for correct pairing
    # -e: Multiple patterns to match
    # This ensures we ONLY get the specific predictor log lines, not debug/info lines
    cmd = ['grep', '-Hn', '-e', features_pattern, '-e', result_pattern] + log_files

    result = subprocess.run(cmd, capture_output=True, text=True)

    if not result.stdout:
        print(f"  No {algorithm} logs found")
        return []

    # Count lines for progress indication
    total_lines = result.stdout.count('\n')
    print(f"  Processing {total_lines} matching lines...")

    # Process grep output efficiently
    # Format: filename:linenum:<ALGORITHM>_FEATURES: key1=value1 ...
    entries_by_file = {}
    lines_processed = 0
    files_seen = set()

    for line in result.stdout.split('\n'):
        if not line:
            continue

        lines_processed += 1

        # Progress update every 10000 lines
        if lines_processed % 10000 == 0:
            print(f"  Progress: {lines_processed}/{total_lines} lines, {len(files_seen)} files", end='\r')

        # Split on first two colons to get filename, linenum, and content
        # The rest of the line after the second colon is the log content
        parts = line.split(':', 2)
        if len(parts) < 3:
            continue

        filename = os.path.basename(parts[0])
        linenum = int(parts[1])
        content = parts[2]  # This includes everything after linenum

        if filename not in entries_by_file:
            entries_by_file[filename] = {'features': [], 'results': []}
            files_seen.add(filename)

        # Double-check pattern match (grep already filtered, but be extra safe)
        # This ensures we ONLY process lines with the exact patterns we want
        if features_pattern in content:
            # Parse features - only if pattern is present
            features = parse_key_value_line(content, features_pattern)
            if features:  # Only add if parsing succeeded
                entries_by_file[filename]['features'].append((linenum, features))
        elif result_pattern in content:
            # Parse results - only if pattern is present
            results = parse_key_value_line(content, result_pattern)
            if results:  # Only add if parsing succeeded
                entries_by_file[filename]['results'].append((linenum, results))

    # Clear progress line
    if lines_processed > 0:
        print(f"  Processed {lines_processed} lines from {len(files_seen)} files    ")

    # Match features with results (pair them in order by line number)
    print(f"  Pairing features with results...")
    entries = []
    files_processed = 0
    total_files = len(entries_by_file)

    for filename, data in entries_by_file.items():
        files_processed += 1

        # Progress update every 10 files
        if files_processed % 10 == 0 or files_processed == total_files:
            print(f"  Pairing: {files_processed}/{total_files} files, {len(entries)} entries found", end='\r')

        features_list = sorted(data['features'])  # Sort by line number
        results_list = sorted(data['results'])

        # Pair up features and results in order
        for i, (_, features) in enumerate(features_list):
            if i < len(results_list):
                _, results = results_list[i]

                # Create combined entry
                entry = {'file': filename}
                entry.update(features)
                entry.update(results)

                # Rename 'iterations' to 'iter' for consistency with train_regressor.py
                if 'iterations' in entry:
                    entry['iter'] = entry.pop('iterations')

                entries.append(entry)

    # Clear progress line and show final count
    if total_files > 0:
        print(f"  Found {len(entries)} complete entries from {total_files} files       ")

    return entries


# Algorithm-specific wrappers for the generic parser
# These provide a clean API and eliminate code duplication

def parse_fp_logs(log_files: List[str]) -> List[Dict[str, Any]]:
    """Parse Feasibility Pump feature and result logs."""
    return parse_generic_algorithm_logs(log_files, 'FP', 'Feasibility Pump')


def parse_pdlp_logs(log_files: List[str]) -> List[Dict[str, Any]]:
    """Parse PDLP (LP Solver) feature and result logs."""
    return parse_generic_algorithm_logs(log_files, 'PDLP', 'LP Solver')


def parse_cp_logs(log_files: List[str]) -> List[Dict[str, Any]]:
    """Parse Constraint Propagation feature and result logs."""
    return parse_generic_algorithm_logs(log_files, 'CP', 'Constraint Propagation')


def parse_fj_logs(log_files: List[str]) -> List[Dict[str, Any]]:
    """
    Parse legacy Feasibility Jump logs (original format).

    Parses lines containing "FJ:" with key=value pairs.
    Uses grep efficiently to minimize Python-side processing.
    """
    print("\nParsing FJ (Feasibility Jump) legacy logs...")
    print(f"  Running grep on {len(log_files)} files...")

    # Use grep to efficiently extract ONLY lines with the exact "FJ:" pattern
    # Note: FJ uses legacy format with just "FJ:" prefix (not FJ_FEATURES/FJ_RESULT)
    # The colon ensures we don't match other FJ-related debug lines
    cmd = ['grep', '-H', 'FJ:'] + log_files
    result = subprocess.run(cmd, capture_output=True, text=True)

    if not result.stdout:
        print(f"  No FJ logs found")
        return []

    # Count lines for progress indication
    total_lines = result.stdout.count('\n')
    print(f"  Processing {total_lines} matching lines...")

    # Parse grep output efficiently
    entries = []
    lines_processed = 0

    for line in result.stdout.split('\n'):
        if not line:
            continue

        lines_processed += 1

        # Progress update every 10000 lines
        if lines_processed % 10000 == 0:
            print(f"  Progress: {lines_processed}/{total_lines} lines, {len(entries)} entries", end='\r')

        # Grep output format: filename:FJ: key1=value1 key2=value2 ...
        parts = line.split(':', 2)
        if len(parts) < 3:
            continue

        filename = os.path.basename(parts[0])
        content = parts[2]

        # Remove "FJ:" prefix if present
        if content.startswith('FJ:'):
            content = content[3:].strip()

        # Parse key-value pairs
        entry = {'file': filename}
        for kv_pair in content.split():
            if '=' in kv_pair:
                key, value = kv_pair.split('=', 1)
                entry[key] = parse_value(value)

        # Only add entry if it has more than just the filename
        if len(entry) > 1:
            entries.append(entry)

    # Clear progress line
    if lines_processed > 0:
        print(f"  Found {len(entries)} entries from {total_lines} lines       ")

    return entries


def print_statistics(entries: List[Dict[str, Any]], algorithm: str) -> None:
    """Print statistics about parsed entries."""
    if not entries:
        print(f"\n  No entries found for {algorithm}")
        return

    unique_files = set(entry['file'] for entry in entries)
    avg_entries_per_file = len(entries) / len(unique_files) if unique_files else 0

    # Check if 'iter' field exists
    has_iter = all('iter' in entry for entry in entries)

    if has_iter:
        iter_values = [entry['iter'] for entry in entries]
        min_iter = min(iter_values)
        max_iter = max(iter_values)
        avg_iter = sum(iter_values) / len(iter_values)

        print(f"\n  Total entries: {len(entries)}")
        print(f"  Unique files: {len(unique_files)}")
        print(f"  Avg entries per file: {avg_entries_per_file:.2f}")
        print(f"  Iterations (target): min={min_iter}, max={max_iter}, avg={avg_iter:.2f}")
    else:
        print(f"\n  Total entries: {len(entries)}")
        print(f"  Unique files: {len(unique_files)}")
        print(f"  Avg entries per file: {avg_entries_per_file:.2f}")

    # Show sample entry
    if entries:
        print(f"\n  Sample entry (first):")
        sample = entries[0]
        for key, value in sorted(sample.items()):
            print(f"    {key}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description='Parse algorithm feature logs and export to pickle for training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Supported Algorithms:
  FP    - Feasibility Pump (parses FP_FEATURES and FP_RESULT logs)
  PDLP  - LP Solver (parses PDLP_FEATURES and PDLP_RESULT logs)
  CP    - Constraint Propagation (parses CP_FEATURES and CP_RESULT logs)
  FJ    - Feasibility Jump (parses legacy FJ: format)

Examples:
  python determinism_logs_parse.py logs/ --algorithm FP -o fp_data.pkl
  python determinism_logs_parse.py logs/ --algorithm PDLP -o pdlp_data.pkl
  python determinism_logs_parse.py logs/ --algorithm CP -o cp_data.pkl
  python determinism_logs_parse.py logs/ --algorithm FJ -o fj_data.pkl
        """
    )

    parser.add_argument(
        'input_dir',
        help='Directory containing .log files to parse'
    )
    parser.add_argument(
        '--algorithm', '-a',
        required=True,
        choices=SUPPORTED_ALGORITHMS,
        help='Algorithm to parse logs for'
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output pickle file path (default: <algorithm>_data.pkl)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print verbose output including warnings'
    )

    args = parser.parse_args()

    # Set default output filename based on algorithm
    if args.output is None:
        args.output = f"{args.algorithm.lower()}_data.pkl"

    # Find all .log files in the input directory
    print(f"\nScanning {args.input_dir} for .log files...")
    log_files = glob.glob(os.path.join(args.input_dir, '*.log'))

    if not log_files:
        print(f"Error: No .log files found in {args.input_dir}")
        return 1

    print(f"Found {len(log_files)} log files")

    # Parse logs based on algorithm
    if args.algorithm == 'FP':
        entries = parse_fp_logs(log_files)
    elif args.algorithm == 'PDLP':
        entries = parse_pdlp_logs(log_files)
    elif args.algorithm == 'CP':
        entries = parse_cp_logs(log_files)
    elif args.algorithm == 'FJ':
        entries = parse_fj_logs(log_files)
    else:
        print(f"Error: Unsupported algorithm: {args.algorithm}")
        return 1

    if not entries:
        print(f"\nError: No entries found for {args.algorithm}")
        print(f"Make sure your logs contain {args.algorithm}_FEATURES and {args.algorithm}_RESULT lines")
        return 1

    # Print statistics
    print_statistics(entries, args.algorithm)

    # Save to pickle file
    print(f"\nSaving {len(entries)} entries to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(entries, f)

    # Get file size
    file_size_bytes = os.path.getsize(args.output)
    file_size_mb = file_size_bytes / (1024 * 1024)

    print(f"\n{'='*70}")
    print(f"✓ Success! Saved {len(entries)} entries to {args.output}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"{'='*70}")
    print(f"\nNext steps:")
    print(f"  1. View available features:")
    print(f"     python scripts/train_regressor.py {args.output} --regressor xgboost --list-features")
    print(f"  2. Train a model:")
    print(f"     python scripts/train_regressor.py {args.output} --regressor xgboost --seed 42")
    print(f"  3. Train with early stopping and C++ export:")
    print(f"     python scripts/train_regressor.py {args.output} --regressor xgboost --seed 42 --early-stopping 20 --treelite-compile 8")

    return 0


if __name__ == '__main__':
    exit(main())
