"""
Example: Solve a BILP from MPS file using reduced cost fixing

This script demonstrates how to load a large Binary Integer Linear Program
from an MPS file and solve it using the two-stage PDLP + reduced cost fixing approach.

Usage:
    python example_solve_mps.py my_problem.mps [tolerance]

Arguments:
    my_problem.mps : Path to MPS file (required)
    tolerance      : Fixing tolerance, default=1e-5 (optional)

Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

import sys
from bilp_reduced_cost_solver import BILPReducedCostSolver
from cuopt.linear_programming.solver_settings import SolverSettings


def main():
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python example_solve_mps.py <mps_file> [tolerance]")
        print("Example: python example_solve_mps.py my_bilp.mps 1e-5")
        sys.exit(1)

    mps_file = sys.argv[1]
    tolerance = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-5

    print("=" * 70)
    print("BILP Solver with PDLP Root Node and Reduced Cost Fixing")
    print("=" * 70)
    print(f"MPS File: {mps_file}")
    print(f"Fixing Tolerance: {tolerance:.2e}")
    print("=" * 70 + "\n")

    # Create solver
    solver = BILPReducedCostSolver(fixing_tolerance=tolerance, verbose=True)

    # Configure settings for large problems
    mip_settings = SolverSettings()
    mip_settings.set_parameter("time_limit", 3600.0)  # 1 hour
    mip_settings.set_parameter("mip_relative_gap", 1e-4)
    mip_settings.set_parameter("mip_absolute_gap", 1e-6)
    mip_settings.set_parameter("log_to_console", True)

    # Solve
    solution = solver.solve_mps(mps_file, mip_settings)

    # Display results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Termination Status: {solution.get_termination_status()}")
    print(f"Objective Value: {solution.get_objective_value()}")

    # Get statistics
    stats = solver.get_statistics()
    print(f"\nSolver Statistics:")
    print(f"  LP Relaxation Bound: {stats['lp_objective']:.16e}")
    print(f"  Variables Fixed to 0: {stats['n_fixed_to_zero']:,}")
    print(f"  Variables Fixed to 1: {stats['n_fixed_to_one']:,}")
    print(f"  Total Variables Fixed: {stats['n_fixed_to_zero'] + stats['n_fixed_to_one']:,}")
    print(f"  Problem Size Reduction: {stats['reduction_percentage']:.2f}%")

    # Compute gap
    if solution.get_objective_value() is not None and stats['lp_objective'] is not None:
        obj_val = solution.get_objective_value()
        lp_bound = stats['lp_objective']
        gap = abs(obj_val - lp_bound)
        rel_gap = gap / (abs(lp_bound) + 1e-10) * 100
        print(f"\nOptimality Gap:")
        print(f"  Absolute Gap: {gap:.6e}")
        print(f"  Relative Gap: {rel_gap:.4f}%")

    # Get solve time
    solve_time = solution.get_solve_time()
    print(f"\nTotal Solve Time: {solve_time:.2f} seconds")

    # Sample solution values (first 20 variables)
    primal_solution = solution.get_primal_solution()
    if len(primal_solution) > 0:
        n_to_show = min(20, len(primal_solution))
        print(f"\nSample Variable Values (first {n_to_show}):")
        for i in range(n_to_show):
            print(f"  x[{i}] = {primal_solution[i]:.6f}")

    # Count binary variables at 0 and 1
    if len(primal_solution) > 0:
        n_at_zero = sum(1 for x in primal_solution if abs(x) < 1e-6)
        n_at_one = sum(1 for x in primal_solution if abs(x - 1.0) < 1e-6)
        n_fractional = len(primal_solution) - n_at_zero - n_at_one

        print(f"\nSolution Composition:")
        print(f"  Variables at 0: {n_at_zero:,} ({100*n_at_zero/len(primal_solution):.1f}%)")
        print(f"  Variables at 1: {n_at_one:,} ({100*n_at_one/len(primal_solution):.1f}%)")
        print(f"  Fractional: {n_fractional:,} ({100*n_fractional/len(primal_solution):.1f}%)")

    print("=" * 70)
    print("Solve complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
