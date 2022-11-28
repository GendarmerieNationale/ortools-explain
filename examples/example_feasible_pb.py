"""
This example shows how models are created with multiple objectives and how to use local explainability
"""

import timeit
from math import sqrt

from ortools_explain.advanced_solving.log_callback import LogCallback
from ortools_explain.model import SuperModel
from ortools_explain.solver import SuperSolver
from ortools_explain.status import Status


def create_sudoku(model, size_sudoku):
    """We define the sudoku constraints in our model
    We return x which is the set of variables"""
    # We define model variables as we would in the standard OR Tools module
    x = {(i, j, v): model.NewBoolVar("x_%d_%d_%d" % (i, j, v))
         for i in range(size_sudoku) for j in range(size_sudoku)
         for v in range(1, size_sudoku + 1)}

    # We define the constraints within a sudoku grid
    # 0- There must be one and only one number by square
    # This one we do not assign a general type to, because we consider it to be a fundamental constraint, one we should never release
    # It is a "background block"
    for i in range(size_sudoku):
        for j in range(size_sudoku):
            model.Add(sum(x[i, j, v] for v in range(1, size_sudoku + 1)) == 1)

    # 1- One of each number by line
    # We define this constraint with general type "all_different_line" and dimensions "line" and value
    for i in range(size_sudoku):
        for v in range(1, size_sudoku + 1):
            model.Add(sum(x[i, j, v] for j in range(size_sudoku)) == 1, "all_different_line", line=i + 1, value=v)

    model.AddExplanation("all_different_line", "All lines must contain each value once and only once",
                         "{line}th line must contain each value once and only once",
                         "All lines must contain one and only one {value}",
                         "{line}th line must contain one and only one {value}")

    # 2- One of each number by column
    # Same as line, but for the sake of example, here we decide to only declare dimension "column"
    for j in range(size_sudoku):
        for v in range(1, size_sudoku + 1):
            model.Add(sum(x[i, j, v] for i in range(size_sudoku)) == 1, "all_different_column", column=j + 1)

    model.AddExplanation("all_different_column", "All columns must contain each value once and only once",
                         "{column}th column must contain each value once and only once")

    # 3- One of each number by square
    k = int(sqrt(size_sudoku))
    for a in range(k):
        for b in range(k):
            for v in range(1, size_sudoku + 1):
                list_cells = [x[i, j, v] for i in range(a * k, (a + 1) * k)
                              for j in range(b * k, (b + 1) * k)]
                model.Add(sum(x for x in list_cells) == 1, "all_different_square", square=str(a + 1) + "_" + str(b + 1))

    model.AddExplanation("all_different_square", "All squares must contain each value once and only once",
                         "Square {square} must contain each value once and only once")

    return x


def add_objectives(model, size_sudoku, x):
    """We add some random objectives to our sudoku:

    * At rank 1:
    ** 9 objectives: "Put a 1 on this square" for each square of the first diagonal
    * At rank 2:
    ** Maximize the sum of the first diagonal
    * At rank 3:
    ** Minimize the sum of the second diagonal
    ** Try putting a 9 on the top right square

    The objective value at rank 3 is (30 if there is a 9 on the top right square, 0 if there is not)
    minus twice the sum of the second diagonal

    """
    for i in range(size_sudoku):
        model.AddRelaxableConstraint(x[i, i, 1] == 1, idx="Numbers of 1 on first diagonal", coef=1, priority=1)

    model.AddMaximumObjective(sum(v * x[i, i, v] for i in range(size_sudoku) for v in range(1, size_sudoku + 1)), priority=2,
                              idx='Maximize sum of first diagonal')

    model.AddMinimumObjective(2 * sum(v * x[i, size_sudoku - 1 - i, v] for i in range(size_sudoku) for v in range(1, size_sudoku + 1)),
                              priority=3, idx='Minimize sum of second diagonal')

    model.AddRelaxableConstraint(x[0, 8, 9] == 1, idx="Top left square should be a 9", coef=30, priority=3)


def write_result(solver, size_sudoku, x):
    tab_val = dict()
    for i in range(size_sudoku):
        for j in range(size_sudoku):
            tab_val[i, j] = '.'
            for v in range(1, size_sudoku+1):
                if solver.Value(x[i, j, v]):
                    tab_val[i, j] = v
                    break
    n = size_sudoku
    k = int(sqrt(n))
    for i in range(n):
        for j in range(n):
            if tab_val[i, j]:
                print(tab_val[i, j], end=" ")
            else:
                print(".", end=" ")
            if j + 1 < n and (j + 1) % k == 0:
                print("|", end=" ")
        print()
        if i + 1 < n and (i + 1) % k == 0:
            print("- " * (n + 2))


if __name__ == "__main__":

    start_time = timeit.default_timer()
    list_background = []

    # --- CREATION OF SUDOKU (EMPTY GRID) ---

    model = SuperModel()
    size_sudoku = 9
    x = create_sudoku(model, size_sudoku)

    # --- ADDING SOME OBJECTIVES TO THE SUDOKU ---
    add_objectives(model, size_sudoku, x)

    print("Model was created in {} seconds".format(round(timeit.default_timer() - start_time, 3)))
    print()

    # --- CREATING A CALLBACK METHOD ---

    my_callback = LogCallback(model, "/home/sarah.petroff/PycharmProjects/ex_json_output.json")

    # --- SOLVING ---

    start_time = timeit.default_timer()

    my_solver = SuperSolver(model)
    status = my_solver.Solve(solution_callback=my_callback)

    if status == Status.FEASIBLE or status == Status.OPTIMAL:
        print("A solution was found in {} seconds: ".format(round(timeit.default_timer() - start_time, 3)))
        write_result(my_solver, size_sudoku, x)
        my_obj = my_solver.GetObjectiveValues()
        print()
        print(my_obj)
        print()
        print('-----')
        print()

        # --- LOCAL EXPLANATION ---

        # Explain why there is a 1 in the top left corner
        start_time = timeit.default_timer()
        print('Explaining {}: '.format(x[0, 0, 1]))
        explanation = my_solver.ExplainValueOfVar(x[0, 0, 1])
        print("An explanation was found in {} seconds: ".format(round(timeit.default_timer() - start_time, 3)))
        print(explanation)
        print()
        print('-----')
        print()

        # Explain why there is a 9 in the top right corner
        start_time = timeit.default_timer()
        print('Explaining {}: '.format(x[0, 8, 9]))
        explanation = my_solver.ExplainValueOfVar(x[0, 8, 9])
        print("An explanation was found in {} seconds: ".format(round(timeit.default_timer() - start_time, 3)))
        print(explanation)
        print()
        print('-----')
        print()

        # Explain why the sum of the first diagonal is higher than the sum of the second diagonal
        start_time = timeit.default_timer()
        print('Explaining why the sum of the first diagonal is higher than the sum of the second diagonal:')
        explanation = my_solver.ExplainWhyNot(sum(v * x[i, i, v] for i in range(size_sudoku) for v in range(1, size_sudoku + 1)) <=
                                              sum(v * x[i, size_sudoku - 1 - i, v] for i in range(size_sudoku) for v in range(1, size_sudoku + 1)))
        print("An explanation was found in {} seconds: ".format(round(timeit.default_timer() - start_time, 3)))
        print(explanation)
        print()
        print('-----')
        print()

        # Explain why the sum of the first line is 45
        start_time = timeit.default_timer()
        print('Explaining why the sum of the first line is 45:')
        explanation = my_solver.ExplainWhyNot(sum(v * x[0, j, v] for j in range(size_sudoku) for v in range(1, size_sudoku + 1)) != 45)
        print("An explanation was found in {} seconds: ".format(round(timeit.default_timer() - start_time, 3)))
        print(explanation)
        print()
        print('-----')
        print()
