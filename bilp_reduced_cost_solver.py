"""
Two-Stage BILP Solver using PDLP Root Node with Reduced Cost Fixing

A Python-only implementation using cuOpt's Python API.
Specialized for large Binary Integer Linear Programs (BILP) with maximization objectives.

Usage:
    from bilp_reduced_cost_solver import BILPReducedCostSolver
    from cuopt.linear_programming import Problem, SolverSettings, MAXIMIZE

    # Create your BILP problem
    problem = Problem(n_constraints=m, n_variables=n)
    # ... set up problem ...

    # Solve with reduced cost fixing
    solver = BILPReducedCostSolver(fixing_tolerance=1e-5)
    solution = solver.solve(problem, mip_settings)

Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

import numpy as np
import copy
import cuopt_mps_parser
from cuopt.linear_programming import Problem
from cuopt.linear_programming.solver_settings import SolverSettings, SolverMethod
from cuopt.linear_programming.solver import Solve


class BILPReducedCostSolver:
    """
    Two-stage solver for large Binary Integer Linear Programs.

    Algorithm:
    1. Solve LP relaxation at root node using PDLP only (no crossover)
    2. Fix variables based on reduced costs and tolerance
    3. Solve reduced BILP with cuOpt's MILP solver

    Parameters
    ----------
    fixing_tolerance : float, default=1e-5
        Tolerance for fixing variables. Variables with values <= tolerance
        or >= 1-tolerance may be fixed based on reduced costs.
    verbose : bool, default=True
        Print progress information during solve.

    Attributes
    ----------
    lp_objective : float
        Objective value of the LP relaxation (after stage 1)
    n_fixed_to_zero : int
        Number of variables fixed to 0 (after stage 2)
    n_fixed_to_one : int
        Number of variables fixed to 1 (after stage 2)
    reduction_percentage : float
        Percentage of variables that were fixed
    """

    def __init__(self, fixing_tolerance=1e-5, verbose=True):
        self.fixing_tolerance = fixing_tolerance
        self.verbose = verbose

        # Statistics
        self.lp_objective = None
        self.n_fixed_to_zero = 0
        self.n_fixed_to_one = 0
        self.reduction_percentage = 0.0

    def _log(self, message):
        """Print message if verbose is enabled."""
        if self.verbose:
            print(message)

    def solve(self, problem, mip_settings=None):
        """
        Solve a Binary Integer Linear Program using two-stage approach.

        Parameters
        ----------
        problem : Problem
            The BILP problem to solve (should be maximization with binary variables)
        mip_settings : SolverSettings, optional
            Settings for the final MILP solve. If None, default settings are used.

        Returns
        -------
        solution
            The solution object from cuOpt's MIP solver

        Notes
        -----
        This method modifies the problem's variable bounds in-place during stage 2.
        The original problem is not preserved.
        """

        if mip_settings is None:
            mip_settings = SolverSettings()

        self._log("=" * 70)
        self._log("Two-Stage BILP Solver: PDLP + Reduced Cost Fixing")
        self._log("=" * 70)

        # Get problem info from the data model
        problem._to_data_model()
        data_model = problem.model
        n_vars = data_model.nVars
        n_cons = data_model.nConstrs

        self._log(f"Problem size: {n_vars} variables, {n_cons} constraints")
        self._log(f"Fixing tolerance: {self.fixing_tolerance:.2e}\n")

        # ====================================================================
        # STAGE 1: Solve LP relaxation with PDLP only (no crossover)
        # ====================================================================
        self._log("Stage 1: Solving LP relaxation with PDLP (no crossover)")

        # Create LP relaxation by removing integer constraints
        original_var_types = copy.deepcopy(data_model.vType)
        data_model.vType = ['C'] * n_vars  # All continuous

        # Configure PDLP settings for root node
        lp_settings = SolverSettings()
        lp_settings.set_parameter("method", SolverMethod.PDLP)  # PDLP only
        lp_settings.set_parameter("crossover", False)  # NO crossover
        lp_settings.set_parameter("time_limit", mip_settings.get_parameter("time_limit"))
        lp_settings.set_parameter("log_to_console", self.verbose)

        # Solve LP relaxation
        lp_solution = Solve(data_model, lp_settings)

        # Check if LP solved successfully
        lp_status = lp_solution.get_termination_status()
        if lp_status != "Optimal":
            self._log(f"Warning: LP relaxation status = {lp_status}")
            self._log("Falling back to regular MIP solve without fixing\n")

            # Restore integer constraints
            data_model.vType = original_var_types
            problem.model = data_model

            # Solve as regular MIP
            return Solve(data_model, mip_settings)

        self.lp_objective = lp_solution.get_objective_value()
        self._log(f"LP relaxation solved, objective: {self.lp_objective:.16e}\n")

        # ====================================================================
        # STAGE 2: Fix variables based on reduced costs
        # ====================================================================
        self._log("Stage 2: Fixing variables based on reduced costs")

        # Get primal solution and reduced costs from LP
        primal_solution = lp_solution.get_primal_solution()
        reduced_costs = lp_solution.get_reduced_cost()

        if reduced_costs is None or len(reduced_costs) == 0:
            self._log("Warning: No reduced costs available, skipping fixing")
            data_model.vType = original_var_types
            problem.model = data_model
            return Solve(data_model, mip_settings)

        # Get current variable bounds
        var_lb = np.array(data_model.lb)
        var_ub = np.array(data_model.ub)

        self.n_fixed_to_zero = 0
        self.n_fixed_to_one = 0

        for i in range(n_vars):
            value = primal_solution[i]
            reduced_cost = reduced_costs[i]

            # For MAXIMIZATION problems:
            # - Negative reduced cost + value near 0 → fix to 0
            # - Positive reduced cost + value near 1 → fix to 1

            if value <= self.fixing_tolerance and reduced_cost < 0.0:
                # Fix variable to 0
                var_lb[i] = 0.0
                var_ub[i] = 0.0
                self.n_fixed_to_zero += 1

            elif value >= (1.0 - self.fixing_tolerance) and reduced_cost > 0.0:
                # Fix variable to 1
                var_lb[i] = 1.0
                var_ub[i] = 1.0
                self.n_fixed_to_one += 1

        n_free = n_vars - self.n_fixed_to_zero - self.n_fixed_to_one
        self.reduction_percentage = 100.0 * (1.0 - n_free / n_vars)

        self._log(f"Variables fixed: {self.n_fixed_to_zero} to zero, {self.n_fixed_to_one} to one")
        self._log(f"Variables remaining free: {n_free}")
        self._log(f"Problem size reduction: {self.reduction_percentage:.1f}%\n")

        # Update problem bounds
        data_model.lb = var_lb.tolist()
        data_model.ub = var_ub.tolist()

        # Restore integer constraints
        data_model.vType = original_var_types
        problem.model = data_model

        # ====================================================================
        # STAGE 3: Solve reduced BILP with MILP solver
        # ====================================================================
        self._log("Stage 3: Solving reduced BILP with MILP solver")
        self._log("=" * 70 + "\n")

        mip_solution = Solve(data_model, mip_settings)

        self._log("\n" + "=" * 70)
        self._log("Two-stage BILP solver completed")
        self._log(f"Final objective: {mip_solution.get_objective_value():.16e}")
        self._log(f"LP relaxation bound: {self.lp_objective:.16e}")

        if mip_solution.get_objective_value() is not None and self.lp_objective is not None:
            gap = abs(mip_solution.get_objective_value() - self.lp_objective)
            rel_gap = gap / (abs(self.lp_objective) + 1e-10) * 100
            self._log(f"Optimality gap: {gap:.6e} ({rel_gap:.4f}%)")

        self._log("=" * 70)

        return mip_solution

    def solve_mps(self, mps_file_path, mip_settings=None):
        """
        Solve a Binary Integer Linear Program from an MPS file using two-stage approach.

        Parameters
        ----------
        mps_file_path : str
            Path to the MPS file containing the BILP problem
        mip_settings : SolverSettings, optional
            Settings for the final MILP solve. If None, default settings are used.

        Returns
        -------
        solution
            The solution object from cuOpt's MIP solver

        Examples
        --------
        >>> solver = BILPReducedCostSolver(fixing_tolerance=1e-5)
        >>> solution = solver.solve_mps("my_large_bilp.mps")
        >>> print(f"Objective: {solution.get_objective_value()}")
        """

        if mip_settings is None:
            mip_settings = SolverSettings()

        self._log("=" * 70)
        self._log("Two-Stage BILP Solver: PDLP + Reduced Cost Fixing")
        self._log("=" * 70)
        self._log(f"Loading MPS file: {mps_file_path}\n")

        # Parse MPS file
        data_model = cuopt_mps_parser.ParseMps(mps_file_path)

        # Get problem dimensions from data model
        n_vars = len(data_model.get_objective_coefficients())
        n_cons = len(data_model.get_constraint_matrix_offsets()) - 1

        self._log(f"Problem size: {n_vars} variables, {n_cons} constraints")
        self._log(f"Fixing tolerance: {self.fixing_tolerance:.2e}\n")

        # ====================================================================
        # STAGE 1: Solve LP relaxation with PDLP only (no crossover)
        # ====================================================================
        self._log("Stage 1: Solving LP relaxation with PDLP (no crossover)")

        # Parse MPS file again for LP relaxation (we'll solve as LP by not treating as MIP)
        lp_data_model = cuopt_mps_parser.ParseMps(mps_file_path)

        # Configure PDLP settings for root node - treat as LP (no MIP)
        lp_settings = SolverSettings()
        lp_settings.set_parameter("method", SolverMethod.PDLP)  # PDLP only
        lp_settings.set_parameter("crossover", False)  # NO crossover
        lp_settings.set_parameter("time_limit", mip_settings.get_parameter("time_limit"))
        lp_settings.set_parameter("log_to_console", self.verbose)

        # Solve LP relaxation (Solve function will treat it as LP if we don't request MIP)
        lp_solution = Solve(lp_data_model, lp_settings)

        # Check if LP solved successfully
        lp_status = lp_solution.get_termination_status()
        if lp_status != "Optimal":
            self._log(f"Warning: LP relaxation status = {lp_status}")
            self._log("Falling back to regular MIP solve without fixing\n")
            return Solve(data_model, mip_settings)

        self.lp_objective = lp_solution.get_objective_value()
        self._log(f"LP relaxation solved, objective: {self.lp_objective:.16e}\n")

        # ====================================================================
        # STAGE 2: Fix variables based on reduced costs
        # ====================================================================
        self._log("Stage 2: Fixing variables based on reduced costs")

        # Get primal solution and reduced costs from LP
        primal_solution = lp_solution.get_primal_solution()
        reduced_costs = lp_solution.get_reduced_cost()

        if reduced_costs is None or len(reduced_costs) == 0:
            self._log("Warning: No reduced costs available, skipping fixing")
            return Solve(data_model, mip_settings)

        # Get current variable bounds
        var_lb = np.array(data_model.get_variable_lower_bounds())
        var_ub = np.array(data_model.get_variable_upper_bounds())

        self.n_fixed_to_zero = 0
        self.n_fixed_to_one = 0

        for i in range(n_vars):
            value = primal_solution[i]
            reduced_cost = reduced_costs[i]

            # For MAXIMIZATION problems:
            # - Negative reduced cost + value near 0 → fix to 0
            # - Positive reduced cost + value near 1 → fix to 1

            if value <= self.fixing_tolerance and reduced_cost < 0.0:
                # Fix variable to 0
                var_lb[i] = 0.0
                var_ub[i] = 0.0
                self.n_fixed_to_zero += 1

            elif value >= (1.0 - self.fixing_tolerance) and reduced_cost > 0.0:
                # Fix variable to 1
                var_lb[i] = 1.0
                var_ub[i] = 1.0
                self.n_fixed_to_one += 1

        n_free = n_vars - self.n_fixed_to_zero - self.n_fixed_to_one
        self.reduction_percentage = 100.0 * (1.0 - n_free / n_vars)

        self._log(f"Variables fixed: {self.n_fixed_to_zero} to zero, {self.n_fixed_to_one} to one")
        self._log(f"Variables remaining free: {n_free}")
        self._log(f"Problem size reduction: {self.reduction_percentage:.1f}%\n")

        # Update problem bounds
        data_model.set_variable_lower_bounds(var_lb.tolist())
        data_model.set_variable_upper_bounds(var_ub.tolist())

        # ====================================================================
        # STAGE 3: Solve reduced BILP with MILP solver
        # ====================================================================
        self._log("Stage 3: Solving reduced BILP with MILP solver")
        self._log("=" * 70 + "\n")

        mip_solution = Solve(data_model, mip_settings)

        self._log("\n" + "=" * 70)
        self._log("Two-stage BILP solver completed")
        self._log(f"Final objective: {mip_solution.get_objective_value():.16e}")
        self._log(f"LP relaxation bound: {self.lp_objective:.16e}")

        if mip_solution.get_objective_value() is not None and self.lp_objective is not None:
            gap = abs(mip_solution.get_objective_value() - self.lp_objective)
            rel_gap = gap / (abs(self.lp_objective) + 1e-10) * 100
            self._log(f"Optimality gap: {gap:.6e} ({rel_gap:.4f}%)")

        self._log("=" * 70)

        return mip_solution

    def get_statistics(self):
        """
        Get statistics about the solving process.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'lp_objective': LP relaxation objective
            - 'n_fixed_to_zero': Number of variables fixed to 0
            - 'n_fixed_to_one': Number of variables fixed to 1
            - 'reduction_percentage': Percentage reduction in problem size
        """
        return {
            'lp_objective': self.lp_objective,
            'n_fixed_to_zero': self.n_fixed_to_zero,
            'n_fixed_to_one': self.n_fixed_to_one,
            'reduction_percentage': self.reduction_percentage
        }


# ============================================================================
# Example Usage
# ============================================================================
if __name__ == "__main__":
    from cuopt.linear_programming import Problem, MAXIMIZE, INTEGER

    print("Creating example Binary Knapsack Problem\n")

    # Simple binary knapsack: maximize value subject to weight constraint
    # max: 10*x1 + 15*x2 + 12*x3 + 8*x4 + 20*x5
    # s.t.: 5*x1 + 7*x2 + 4*x3 + 3*x4 + 8*x5 <= 15
    #       x1, x2, x3, x4, x5 ∈ {0, 1}

    problem = Problem("knapsack_example")

    # Add binary variables
    x1 = problem.addVariable(lb=0.0, ub=1.0, obj=10.0, vtype=INTEGER, vname="x1")
    x2 = problem.addVariable(lb=0.0, ub=1.0, obj=15.0, vtype=INTEGER, vname="x2")
    x3 = problem.addVariable(lb=0.0, ub=1.0, obj=12.0, vtype=INTEGER, vname="x3")
    x4 = problem.addVariable(lb=0.0, ub=1.0, obj=8.0, vtype=INTEGER, vname="x4")
    x5 = problem.addVariable(lb=0.0, ub=1.0, obj=20.0, vtype=INTEGER, vname="x5")

    # Add weight constraint
    problem.addConstraint(5*x1 + 7*x2 + 4*x3 + 3*x4 + 8*x5 <= 15, name="weight")

    # Set objective (maximize)
    problem.setObjective(10*x1 + 15*x2 + 12*x3 + 8*x4 + 20*x5, sense=MAXIMIZE)

    # Create solver settings
    mip_settings = SolverSettings()
    mip_settings.set_parameter("time_limit", 300.0)
    mip_settings.set_parameter("mip_relative_gap", 1e-4)

    # Solve with reduced cost fixing
    solver = BILPReducedCostSolver(fixing_tolerance=1e-5, verbose=True)
    solution = solver.solve(problem, mip_settings)

    # Display results
    print("\n" + "=" * 70)
    print("SOLUTION SUMMARY")
    print("=" * 70)
    print(f"Status: {solution.get_termination_status()}")
    print(f"Objective: {solution.get_objective_value()}")
    print(f"\nVariable values:")
    for var in [x1, x2, x3, x4, x5]:
        print(f"  {var.VariableName} = {var.getValue()}")

    # Display statistics
    stats = solver.get_statistics()
    print(f"\nSolver statistics:")
    print(f"  LP bound: {stats['lp_objective']:.6f}")
    print(f"  Variables fixed to 0: {stats['n_fixed_to_zero']}")
    print(f"  Variables fixed to 1: {stats['n_fixed_to_one']}")
    print(f"  Problem reduction: {stats['reduction_percentage']:.1f}%")
    print("=" * 70)

    # ========================================================================
    # Example 2: Solving from MPS file
    # ========================================================================
    print("\n\n" + "=" * 70)
    print("Example 2: Loading and solving from MPS file")
    print("=" * 70)
    print("\nTo solve from an MPS file:")
    print("""
    solver = BILPReducedCostSolver(fixing_tolerance=1e-5, verbose=True)

    mip_settings = SolverSettings()
    mip_settings.set_parameter("time_limit", 3600.0)
    mip_settings.set_parameter("mip_relative_gap", 1e-4)

    solution = solver.solve_mps("my_large_bilp.mps", mip_settings)

    print(f"Objective: {solution.get_objective_value()}")
    print(f"Status: {solution.get_termination_status()}")

    # Get variable values
    primal = solution.get_primal_solution()
    print(f"First 10 variables: {primal[:10]}")

    # Get statistics
    stats = solver.get_statistics()
    print(f"Problem reduced by {stats['reduction_percentage']:.1f}%")
    """)
