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
from cuopt.linear_programming.solver.solver_parameters import (
    CUOPT_METHOD,
    CUOPT_CROSSOVER,
    CUOPT_PRESOLVE,
    CUOPT_TIME_LIMIT,
    CUOPT_LOG_TO_CONSOLE,
    CUOPT_DUAL_POSTSOLVE,
    CUOPT_ABSOLUTE_PRIMAL_TOLERANCE,
    CUOPT_ABSOLUTE_DUAL_TOLERANCE,
    CUOPT_RELATIVE_PRIMAL_TOLERANCE,
    CUOPT_RELATIVE_DUAL_TOLERANCE
)


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
        lp_settings.set_parameter(CUOPT_METHOD, 1)  # 1 = PDLP
        lp_settings.set_parameter(CUOPT_CROSSOVER, False)  # NO crossover
        lp_settings.set_parameter(CUOPT_PRESOLVE, False)  # DISABLE presolve
        lp_settings.set_parameter(CUOPT_TIME_LIMIT, mip_settings.get_parameter(CUOPT_TIME_LIMIT))
        lp_settings.set_parameter(CUOPT_LOG_TO_CONSOLE, self.verbose)

        # Debug: verify settings
        if self.verbose:
            self._log(f"LP Settings - Method: {lp_settings.get_parameter(CUOPT_METHOD)}")
            self._log(f"LP Settings - Crossover: {lp_settings.get_parameter(CUOPT_CROSSOVER)}")
            self._log(f"LP Settings - Presolve: {lp_settings.get_parameter(CUOPT_PRESOLVE)}")

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

        # Load MIP problem
        mip_problem = Problem.readMPS(mps_file_path)
        n_vars = mip_problem.NumVariables
        n_cons = mip_problem.NumConstraints

        self._log(f"Problem size: {n_vars} variables, {n_cons} constraints")
        self._log(f"Fixing tolerance: {self.fixing_tolerance:.2e}\n")

        # ====================================================================
        # STAGE 1: Solve LP relaxation with PDLP using .relax()
        # ====================================================================
        self._log("Stage 1: Solving LP relaxation with PDLP (no crossover)")

        # Create LP relaxation using .relax() method
        lp_problem = mip_problem.relax()
        self._log("Created LP relaxation using .relax() method")

        # Configure PDLP settings for LP
        lp_settings = SolverSettings()
        lp_settings.set_parameter(CUOPT_METHOD, 1)  # 1 = PDLP
        lp_settings.set_parameter(CUOPT_PRESOLVE, False)  # DISABLE presolve
        lp_settings.set_parameter(CUOPT_CROSSOVER, False)  # NO crossover
        lp_settings.set_parameter(CUOPT_DUAL_POSTSOLVE, False)  # NO dual postsolve

        # Set tight primal-dual gap tolerances (< 1e-7)
        # lp_settings.set_parameter(CUOPT_ABSOLUTE_PRIMAL_TOLERANCE, 1e-8)
        # lp_settings.set_parameter(CUOPT_ABSOLUTE_DUAL_TOLERANCE, 1e-8)
        # lp_settings.set_parameter(CUOPT_RELATIVE_PRIMAL_TOLERANCE, 1e-8)
        # lp_settings.set_parameter(CUOPT_RELATIVE_DUAL_TOLERANCE, 1e-8)

        self._log(f"PDLP settings: METHOD={lp_settings.get_parameter(CUOPT_METHOD)}, "
                  f"CROSSOVER={lp_settings.get_parameter(CUOPT_CROSSOVER)}, "
                  f"PRESOLVE={lp_settings.get_parameter(CUOPT_PRESOLVE)}")
        self._log(f"Tolerances: PRIMAL={lp_settings.get_parameter(CUOPT_ABSOLUTE_PRIMAL_TOLERANCE):.2e}, "
                  f"DUAL={lp_settings.get_parameter(CUOPT_ABSOLUTE_DUAL_TOLERANCE):.2e}")

        lp_settings.set_parameter(CUOPT_TIME_LIMIT, mip_settings.get_parameter(CUOPT_TIME_LIMIT))
        lp_settings.set_parameter(CUOPT_LOG_TO_CONSOLE, self.verbose)

        # Solve LP relaxation
        lp_problem.solve(lp_settings)

        if lp_problem.Status.name != "Optimal":
            self._log(f"Warning: LP relaxation status = {lp_problem.Status.name}")
            self._log("Falling back to regular MIP solve without fixing\n")
            mip_problem.solve(mip_settings)
            return mip_problem

        self.lp_objective = lp_problem.ObjValue
        self._log(f"LP relaxation solved, objective: {self.lp_objective:.16e}\n")

        # ====================================================================
        # STAGE 2: Fix variables based on LP solution values only
        # ====================================================================
        self._log("Stage 2: Fixing variables based on LP solution values")

        # Use tight tolerance (1e-6) for value-based fixing
        value_tolerance = 1e-6
        self._log(f"Fixing criteria (ignoring reduced costs):")
        self._log(f"  Fix to 0 if: value <= {value_tolerance}")
        self._log(f"  Fix to 1 if: value >= {1.0 - value_tolerance}")

        self.n_fixed_to_zero = 0
        self.n_fixed_to_one = 0

        lp_vars = lp_problem.getVariables()
        mip_vars = mip_problem.getVariables()

        # Collect statistics for diagnosis
        values = []
        near_zero_count = 0
        near_one_count = 0

        # First pass: identify candidates for fixing
        fixing_candidates = []  # (var_index, fix_to_value)

        for i, (lp_var, mip_var) in enumerate(zip(lp_vars, mip_vars)):
            value = lp_var.getValue()
            values.append(value)

            # Count variables near bounds
            if value <= value_tolerance:
                near_zero_count += 1
                fixing_candidates.append((i, 0.0))
            elif value >= (1.0 - value_tolerance):
                near_one_count += 1
                fixing_candidates.append((i, 1.0))

        self._log(f"\nInitial candidates: {len(fixing_candidates)} variables")
        self._log(f"  {near_zero_count} near 0, {near_one_count} near 1")

        # Get constraint information to check for tight constraints and avoid infeasibility
        constrs = lp_problem.getConstraints()
        self._log(f"\nAnalyzing {len(constrs)} constraints for safe fixing...")

        # Build variable-to-constraint mapping
        var_to_constraints = {i: [] for i in range(len(lp_vars))}

        for j, constr in enumerate(constrs):
            # Get variables involved in this constraint
            for var in constr.vars:
                var_to_constraints[var.index].append(j)

        # Identify tight constraints
        tight_constraints = set()
        constraint_tolerance = 1e-5  # More relaxed for constraint tightness

        for j, constr in enumerate(constrs):
            slack = abs(constr.Slack) if hasattr(constr, 'Slack') else 0.0
            if slack <= constraint_tolerance:
                tight_constraints.add(j)

        self._log(f"Found {len(tight_constraints)} tight constraints (slack <= {constraint_tolerance})")

        # Second pass: apply fixing with constraint-aware protection
        # Skip fixing variables that appear in tight constraints with small coefficients
        protected_vars = set()

        for j in tight_constraints:
            constr = constrs[j]
            # Check if this constraint has very few free variables
            # If so, we should be careful about fixing variables in it
            free_vars_in_constraint = []
            for var in constr.vars:
                idx = var.index
                val = values[idx]
                if val > value_tolerance and val < (1.0 - value_tolerance):
                    free_vars_in_constraint.append(idx)

            # If constraint has 5 or fewer free variables, protect all variables in it
            if len(free_vars_in_constraint) <= 5:
                for var in constr.vars:
                    protected_vars.add(var.index)

        self._log(f"Protecting {len(protected_vars)} variables involved in critically tight constraints")

        # Apply fixing with protection
        n_skipped = 0
        for i, (lp_var, mip_var) in enumerate(zip(lp_vars, mip_vars)):
            value = lp_var.getValue()

            # Skip protected variables
            if i in protected_vars:
                if value <= value_tolerance or value >= (1.0 - value_tolerance):
                    n_skipped += 1
                continue

            if value <= value_tolerance:
                # Fix variable to 0
                mip_var.setLowerBound(0.0)
                mip_var.setUpperBound(0.0)
                self.n_fixed_to_zero += 1
                if self.n_fixed_to_zero <= 5:  # Show first 5
                    self._log(f"  Var {i}: value={value:.6e} → fixed to 0")

            elif value >= (1.0 - value_tolerance):
                # Fix variable to 1
                mip_var.setLowerBound(1.0)
                mip_var.setUpperBound(1.0)
                self.n_fixed_to_one += 1
                if self.n_fixed_to_one <= 5:  # Show first 5
                    self._log(f"  Var {i}: value={value:.6e} → fixed to 1")

        self._log(f"Skipped fixing {n_skipped} variables to avoid infeasibility")

        # Print diagnostic statistics
        values = np.array(values)

        self._log(f"\nDiagnostic statistics:")
        self._log(f"  Variable values - min: {values.min():.6e}, max: {values.max():.6e}, mean: {values.mean():.6e}")
        self._log(f"  Variables near 0 (≤{value_tolerance}): {near_zero_count}")
        self._log(f"  Variables near 1 (≥{1.0 - value_tolerance}): {near_one_count}")

        n_free = n_vars - self.n_fixed_to_zero - self.n_fixed_to_one
        self.reduction_percentage = 100.0 * (1.0 - n_free / n_vars)

        self._log(f"\nVariables fixed: {self.n_fixed_to_zero} to zero, {self.n_fixed_to_one} to one")
        self._log(f"Variables remaining free: {n_free}")
        self._log(f"Problem size reduction: {self.reduction_percentage:.1f}%\n")

        # ====================================================================
        # STAGE 3: Solve reduced BILP with MILP solver
        # ====================================================================
        self._log("Stage 3: Solving reduced BILP with MILP solver")
        self._log("=" * 70 + "\n")

        mip_problem.solve(mip_settings)

        self._log("\n" + "=" * 70)
        self._log("Two-stage BILP solver completed")
        self._log(f"Final objective: {mip_problem.ObjValue:.16e}")
        self._log(f"LP relaxation bound: {self.lp_objective:.16e}")

        if mip_problem.ObjValue is not None and self.lp_objective is not None:
            gap = abs(mip_problem.ObjValue - self.lp_objective)
            rel_gap = gap / (abs(self.lp_objective) + 1e-10) * 100
            self._log(f"Optimality gap: {gap:.6e} ({rel_gap:.4f}%)")

        self._log("=" * 70)

        return mip_problem

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
