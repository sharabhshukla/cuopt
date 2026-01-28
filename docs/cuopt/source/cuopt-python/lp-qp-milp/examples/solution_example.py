# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION &
# SPDX-License-Identifier: Apache-2.0
# AFFILIATES. All rights reserved.
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
Linear Programming Solution Example

This example demonstrates how to:
- Create a linear programming problem
- Solve the problem and retrieve result details

Problem:
    Minimize: 3x + 2y + 5z
    Subject to:
        x + y + z = 4
        2x + y + z =  5
        x, y, z >= 0

Expected Output:
    Optimal solution found in 0.02 seconds
    Objective: 9.0
    x = 1.0, ReducedCost = 0.0
    y = 3.0, ReducedCost = 0.0
    z = 0.0, ReducedCost = 2.999999858578644
    c1 DualValue = 1.0000000592359144
    c2 DualValue = 1.0000000821854418
"""

from cuopt.linear_programming.problem import Problem, MINIMIZE


def main():
    """Run the simple LP example."""
    problem = Problem("min_dual_rc")

    # Add Variables
    x = problem.addVariable(lb=0.0, name="x")
    y = problem.addVariable(lb=0.0, name="y")
    z = problem.addVariable(lb=0.0, name="z")

    # Add Constraints (equalities)
    problem.addConstraint(x + y + z == 4.0, name="c1")
    problem.addConstraint(2.0 * x + y + z == 5.0, name="c2")

    # Set Objective (minimize)
    problem.setObjective(3.0 * x + 2.0 * y + 5.0 * z, sense=MINIMIZE)

    # Solve
    problem.solve()

    # Check solution status
    if problem.Status.name == "Optimal":
        print(f"Optimal solution found in {problem.SolveTime:.2f} seconds")
        # Get Primal
        print("Objective:", problem.ObjValue)
        for v in problem.getVariables():
            print(
                f"{v.VariableName} = {v.Value}, ReducedCost = {v.ReducedCost}"
            )
        # Get Duals
        for c in problem.getConstraints():
            print(f"{c.ConstraintName} DualValue = {c.DualValue}")
    else:
        print(f"Problem status: {problem.Status.name}")


if __name__ == "__main__":
    main()
