"""
This example shows how models are created and how negative explainability works
"""

import timeit
from math import sqrt

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


def add_initial_positions(model, x, initial_positions):
    for row, col, value in initial_positions:
        model.AddConstant(x[row, col, value], 1, "initial_pos", line=row+1, column=col+1, value=value)

    model.AddExplanation("initial_pos", "Initial positions", "Initial positions on line {line}",
                         "Initial positions on column {column}", "Initial positions of {value}",
                         "Initial position on square ({line}, {column})", "Initial positions of {value} on line {line}",
                         "Initial positions of {value} on column {column}", "The initial {value} at ({line}, {column})")

    # Another possible implementation
    # for row, col, value in initial_positions:
    #     for poss_value in range(1, 5):
    #         if value == poss_value:
    #             model.AddConstant(x[row, col, poss_value], 1, "initial_pos", line=row+1, column=col+1, value=value)
    #         else:
    #             model.AddConstant(x[row, col, poss_value], 0, "initial_pos", line=row+1, column=col+1, value=value)

    # And another one
    # for row, col, value in initial_positions:
    #     model.AddConstant(x[row, col, value], 1, "initial_pos", line=row+1, column=col+1)


def print_sudoku(size_sudoku, initial_positions):
    tab_val = dict()
    for i in range(size_sudoku):
        for j in range(size_sudoku):
            tab_val[i, j] = '.'
            for v in range(1, size_sudoku+1):
                if (i, j, v) in initial_positions:
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


if __name__ == '__main__':
    start_time = timeit.default_timer()
    list_background = []

    # --- CREATION OF SUDOKU (EMPTY GRID) ---
    model = SuperModel()
    size_sudoku = 4
    x = create_sudoku(model, size_sudoku)

    # --- ADDING INITIAL VALUES TO MAKE THE SUDOKU INFEASIBLE ---
    initial_positions = [(0, 0, 1), (0, 1, 1), (2, 0, 3), (3, 0, 3), (3, 1, 3)]
    add_initial_positions(model, x, initial_positions)

    # --- PRINTING THE SUDOKU IN CONSOLE ---
    print_sudoku(4, initial_positions)

    print()
    print("Model was created in {} seconds".format(round(timeit.default_timer() - start_time, 3)))
    print()

    # --- TRYING TO SOLVE

    start_time = timeit.default_timer()

    my_solver = SuperSolver(model)
    status = my_solver.Solve()

    if status == Status.INFEASIBLE:
        print("Model was found infeasible in {} seconds".format(round(timeit.default_timer() - start_time, 3)))

        print()

        # --- EXPLAINING WHY THE PROBLEM IS INFEASIBLE, IN 4 DIFFERENT WAYS

        start_time = timeit.default_timer()
        my_conflicts = my_solver.ExplainWhyNoSolution(method_for_search=SuperSolver.SUFFICIENT_ASSUMPTION, zoom_level=1)
        print('Conflicts found with sufficient assumption: ')
        print(my_conflicts)
        print("Conflicts were found in {} seconds".format(round(timeit.default_timer() - start_time, 3)))
        print()

        start_time = timeit.default_timer()
        my_conflicts = my_solver.ExplainWhyNoSolution(method_for_search=SuperSolver.QUICK_XPLAIN, zoom_level=1)
        print('Conflicts found with quick xplain: ')
        print(my_conflicts)
        print("Conflicts were found in {} seconds".format(round(timeit.default_timer() - start_time, 3)))
        print()

        start_time = timeit.default_timer()
        my_conflicts = my_solver.ExplainWhyNoSolution(method_for_search=SuperSolver.PARALLEL_SEARCH, zoom_level=1)
        print('Conflicts found with parallel search: ')
        print(my_conflicts)
        print("Conflicts were found in {} seconds".format(round(timeit.default_timer() - start_time, 3)))
        print()

        start_time = timeit.default_timer()
        my_conflicts = my_solver.ExplainWhyNoSolution(method_for_search=SuperSolver.PARALLEL_SEARCH, zoom_level=2)
        print('Conflicts found with parallel search and a higher zoom level: ')
        print(my_conflicts)
        print("Conflicts were found in {} seconds".format(round(timeit.default_timer() - start_time, 3)))
        print()
