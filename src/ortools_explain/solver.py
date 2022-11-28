"""
Defines a SuperSolver class that overrides CpSolver and enables you to use complexe optimization strategies and to
produce explanations.
"""
import logging
from typing import Dict, List, Union

from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import IntVar

from .explainability.negative_explanations import NegativeExplanation
from .model import SuperModel
from ortools_explain.model_indexation.objective import MaxObjective, MinObjective, BonusConstraint
from ortools_explain.explainability.positive_explanations import positive_explanations
from ortools_explain.advanced_solving.local_search import LNS_Variables_Choice
from .status import Status

logger = logging.getLogger(__name__)


class SuperSolver(cp_model.CpSolver):
    """
    This class is an overlayer to OR Tools's solver.

    It allows for multi-optimization solving and local positive explainability.

    # Solving with SuperSolver

    SuperSolver mostly works like cp_model.CpSolver but handles multi-objective models.
    my_solver.Solve() returns a status which can be one of the following:

    * UNKNOWN -- similar to cp_model.UNKOWN
    * MODEL_INVALID -- similar to cp_model.MODEL_INVALID
    * FEASIBLE -- similar to cp_model.FEASIBLE
    * INFEASIBLE -- similar to cp_model.INFEASIBLE
    * OPTIMAL -- similar to cp_model.OPTIMAL
    * OBVIOUS_CONFLICT -- the module has not proceeded to launching the solver because an obvious conflict has already been
    detected in the model (for instance a variable has been set to a constant twice with different values)
    * NEVER_LAUNCHED -- the solver has not yet been launched (call *Solve()*)

    ## Example
    ```
    my_model = SuperModel()

    # Create the model here by defining variables, adding constraints and adding objectives

    # WARNING - As opposed to cp_model.CpSolver, SuperSolver takes the model as argument when it is created
    my_solver = SuperSolver(my_model)
    status = my_solver.Solve()  # Solve(model) would also work to mimic CpSolver

    if status == Status.OPTIMAL:
        print("Optimization was successful")
        print(my_solver.GetObjectiveValues())

        # You could do positive explanations here

    elif status == Status.FEASIBLE:
        print("A solution was found but is not optimal")

        # You could do positive explanations here
        # You could do local optimization here

    elif status == Status.OBVIOUS_CONFLICT:
        print("Something is obviously wrong in the model")
        my_conflicts = my_model.list_obvious_conflicts()
        for conflict in my_conflicts:
            print(conflict.write(my_model))

    elif status == Status.INFEASIBLE:
        print("Some conditions are conflicting within the model")
        conflicts = my_solver.ExplainWhyNoSolution()
        print(conflicts)
    ```

    # Local optimization

    SuperSolver contains a LNS function that enables you to improve the solution found by the solver by searching for local improvements.

    To do this, you decide to set part of the problem to its current value and only allow changes in the rest of the problem.

    To use LNS techniques, you must create a class that inherits from the *LNS_Variables_Choice* abstract class.
    Classes inheriting from *LNS_Variables_Choice* need two methods:

    * **variables_to_block()** must return at each call the variables that will not be allowed to change from their current value
    * **nb_remaining_iterations()** must return at each call the number of times to optimize again locally

    ## Example - Scheduling problem

    We consider a scheduling problem where *X[i, j, k]* are boolean variables
    such that *X[i, j, k] == 1* means that person I on day J takes job K.

    ```
    my_model = SuperModel()

    for i in list_people:
        for j in list_days:
            for k in list_missions:
                X[i, j, k] = my_model.NewBoolVar("takes_mission_{}_{}_{}".format(i, j, k)")

    # Constraints are added here
    # Objectives are added here

    my_solver = SuperSolver(my_model)
    ```

    At this point we assume that the solver has reached a solution but this solution is not the very optimum.

    One way to optimize is to decide only to optimize between the 3 persons
    with the worst schedule and the 3 persons with the best schedule, and to do so 10 times in a row:

    ```
    # Here we define our LNS strategy
    def LNS_Equity(LNS_Variables_Choice):
        def __init__(self, X, nb_iterations):
            self.my_variables = X
            self.nb_iterations = nb_iterations
            self.nb_done_optim = 0

        def sort_people_by_quality_of_schedule(self) -> [int]:
            # Return the list of people sorted by increasing order of quality of their schedule in current solution
            pass

        def variables_to_block(self):
            self.nb_done_optim += 1
            people_sorted = self.sort_people_by_quality_of_schedule()
            people_not_to_change = people_sorted[3:-3]
            variables_to_block = [X[i, j, k] for (i, j, k) in product(people_not_to_change, list_days, list_missions)

        def nb_remaining_iterations(self):
            return self.nb_iterations - self.nb_done_optim

    # Here we use it to optimize
    if my_solver.status() == Status.FEASIBLE:
        my_variables = [X[i, j, k] for (i, j, k) in product(list_people, list_days, list_missions)]
        lns_strategy = LNS_Equity(my_variables, nb_iterations= 10)
        my_solver.LNS(lns_strategy, max_time_lns= 300)

    ```

    Another way to optimize is to optimize locally on consecutive days, meaning that we will only focus on 5-day windows and try to optimize them locally.
    We will shift this 5-day window on the whole range of our problem so as to do this everywhere. Here we decide to shift by 2-day shifts.

    If list_days is [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], optimization will thus be done on the following windows:
    [0, 1, 2, 3, 4], [2, 3, 4, 5, 6], [4, 5, 6, 7, 8], [6, 7, 8, 9]

    We provide a built-in function for this specific "one-dimension moving window" type of optimization:

    ```python
    if my_solver.status() == Status.FEASIBLE:
        # We create a dictionary that links each variable to its value on the dimension of interest (here days)
        dict_for_lns = {X[i, j, k]: j for (i, j, k) in product(list_people, list_days, list_missions)}
        # We simply define the LNS strategy with the built-in function
        lns_strategy = LNS_Variables_Choice_Across_One_Dim(dict_for_lns, window_size= 5, step= 2, nb_iterations= 1)

        my_solver.LNS(lns_strategy, max_time_lns= 300)
    ```

    # Local explainability

    The SuperSolver class enables you to ask why one given variable was set to a specific value,
    with *ExplainValueOfVar*. For instance if the solver returns a solution where boolean variable *X[0, 0]* is set to 1,
    you may want to ask why that is. You can use:

    ```
    print(solver.ExplainValueOfVar(X[0, 0]))
    ```

    and the module will return one of the following:

    ```
    # 1- If X[0, 0] is not set to 1, this makes the problem infeasible
    >> {"outcome": "infeasible"}

    # 2- If X[0, 0] is not set to 1, the problem is feasible but you cannot reach the same optimization scores
    >> {"outcome": "less_optimal",
        "optimization_scores": {"old_values": [100, 40, 70],
                                "new_values": [100, 30, 80]},
        "objective_values": [{"id": "sum_x", "old_value": 40, "new_value": 30},
                            {"id": "second_objective", "old_value": 60, "new_value": 60}...
                            ]
        }

    # 3- It is possible to find another solution with the same optimization scores
    # In this case some variables have changed (including obviously X[0, 0]) and the module will return them
    >> {"outcome": "as_optimal",
        "changed_variables": [{"name": x_0_0, "old_value": 1, "new_value": 0},
                             {"name": x_0_1, "old_value": 0, "new_value": 1}...]
        }

    ```

    *ExplainValueOfVar* only enables you to study one variable at a time, but *ExplainWhyNot* enables you
    to ask for more complex explanations:

    ```
    # If in our current solution, the solver has set X[0, 0] to value 1,
    # then the two lines below are identical
    my_explanation = solver.ExplainValueOfVar(X[0, 0])
    my_explanation = solver.ExplainWhyNot(X[0, 0] != 1)

    # ExplainWhyNot allow for more complex explanations
    my_explanation = solver.ExplainWhyNot(sum(X[i, 0] for i in list_I) == 0)
    ```

    """
    PARALLEL_SEARCH = 1,
    QUICK_XPLAIN = 2,
    SUFFICIENT_ASSUMPTION = 3

    def __init__(self, model: SuperModel):
        super().__init__()
        self.model = model
        self.res_optimization = None  # Remembers the objective values
        self._objective_active = True
        self._current_status = Status.NEVER_LAUNCHED
        self.example_best_solution = dict()  # Remembers the solution with the best objective

    def status(self) -> int:
        """
        Returns one the following values

        - UNKNOWN = 0
        - MODEL_INVALID = 1
        - FEASIBLE = 2
        - INFEASIBLE = 3
        - OPTIMAL = 4
        - OBVIOUS_CONFLICT = 5
        - NEVER_LAUNCHED = 6

        """
        return self._current_status

    def _has_a_solution(self) -> bool:
        return self._current_status == Status.OPTIMAL or self._current_status == Status.FEASIBLE

    def LogObjectiveValues(self):
        """Prints the current objective values in the logger."""
        if not self.res_optimization:
            logger.info("Solver has not been launched yet")
            return
        mes_priorites = self.model.objective().get_list_priority()
        nb_priority = len(mes_priorites)
        for i in range(nb_priority):
            rank_prio = i
            val_prio = mes_priorites[i]
            logger.info("Score at rank {}: {}".format(rank_prio, self.res_optimization[rank_prio]))

            for obj in self.model.objective().dic_elt[val_prio]:
                logger.info("-- Value of objective {} is {}".format(obj.get_id(), obj.best_value()))

    def GetObjectiveValues(self) -> Union[None, Dict]:
        """Returns the objective values in a dictionary:
        ```
        {"objective_values":
            {id1: val1,
            id2: val2...
            },
        "score_by_rank": [score1, score2... ]
        }
        ```
        where id1, id2... are partial objectives idx and val1, val2... the value of these objectives,
        and score1, score2... are values achieved at each rank, by increasing order.

        Only works if the model is feasible and the solver has already been launched.
        Otherwise returns None.
        """
        if self._current_status != Status.FEASIBLE and self._current_status != Status.OPTIMAL:
            logger.info('Solver has not yet been launched successfully')
            return None

        res = {"objective_values": {}, "score_by_rank": [self.res_optimization]}
        for obj in self.model.objective().get_all_obj():
            res["objective_values"][obj.get_id()] = obj.best_value()

        return res

    def DisableObjective(self):
        """
        Disables all objectives
        """
        logging.debug("Deactivating objective")
        self._objective_active = False

    def EnableObjective(self):
        """
        Enables all objectives
        """
        self._objective_active = True

    def ExplainValueOfVar(self, var: IntVar) -> Dict:
        """
        Enables you to ask why a variable was set to its current value in the solver solution.

        **var**: The variable you want to get an explanation for

        **See top of the file for examples and additional information**
        """
        val_var = self.Value(var)
        return self.ExplainWhyNot(var != val_var)

    def ExplainWhyNot(self, additional_condition) -> Dict:
        """Enables you to ask why some variables have been set to their current value.

        **additional_condition**: condition that we want to try solving with.

        **See top of the file for examples and additional information**
        """
        return positive_explanations(self, additional_condition)

    def ExplainWhyNoSolution(self, method_for_search: int = PARALLEL_SEARCH, zoom_level: int = 2, find_several_iis: bool = True, split_choice=3) -> List[List[Dict]]:
        """
        Launches a solver for negative explainability depending on parameters passed.

        **Default parameters should fit most cases.**

        **method_for_search**: Method used by the module to look for conflicts, among the following:

        * *SuperSolver.SUFFICIENT_ASSUMPTION*: based on the OR Tools built-in method.
        * *SuperSolver.QUICK_XPLAIN*: implements the Quick Xplain algorithm which is of divide-and-conquer type.
        * *SuperSolver.PARALLEL_SEARCH*: launches the other two methods in parallel threads and returns the fastest one.

        **zoom_level**: When searching for a conflict, the module will at some point try to go deeper in the explanation by splitting one one
        of the constraints' dimensions. When doing this, it will check that the conflict set has not grown by more than *zoom_level* in size.
        Otherwise it will decide that this dimension is not relevant in this specific case and will not split on it.

        **find_several_iis**: If False, the solver will only return the first conflict it encounters. If True, the solver will return a set of not-overlapping conflicts that is maximum in size
        (if you disable all constraints is all conflicts of this set then the problem is feasible).

        **split_choice**: For Quick Xplain, decides on the divide-and-conquer strategy, from a list of constraints. Available values are:

        * 1: the first element of the list is isolated from the other elements
        * 2: the list of constraints is split in half on the number of elements
        * 3: the list of constraints is split in half by trying to balance the size of the two resulting lists
        (elements in the list can encompass from 1 to numerous actual OR-Tools elementary constraints. We use a simple optimization problem to split the
        elements in two different sets with approximately the same size)

        """
        if method_for_search not in [SuperSolver.PARALLEL_SEARCH, SuperSolver.QUICK_XPLAIN, SuperSolver.SUFFICIENT_ASSUMPTION]:
            raise ValueError("{} is not an acceptable method for search".format(method_for_search))

        explainer = NegativeExplanation(self.model, split_choice, nb_iis_granularite=zoom_level)

        if find_several_iis:
            if method_for_search == SuperSolver.PARALLEL_SEARCH:
                res = explainer.find_set_of_distinct_iis_combined()
            elif method_for_search == SuperSolver.QUICK_XPLAIN:
                res = explainer.find_set_of_distinct_iis_qx()
            else:
                res = explainer.find_set_of_distinct_iis_sa()
        else:
            if method_for_search == SuperSolver.PARALLEL_SEARCH:
                res = [explainer.find_one_iis_combined()]
            elif method_for_search == SuperSolver.QUICK_XPLAIN:
                res = [explainer.find_one_iis_auto_zoom_quick_xplain()]
            else:
                res = [explainer.find_one_iis_sufficient_assumption()]

        res_dict = []
        for iis in res:
            iis_dict = []
            for named_block in iis:
                iis_dict.append(named_block.to_dict())
            res_dict.append(iis_dict)
        return res_dict

    def try_solving(self, mute=False):
        """**For internal use only, use Solve() instead**

        Solves the model in several steps, depending on the objective that was defined"""
        if self.model.objective().is_empty():
            return super().Solve(self.model), None, None

        if not self._objective_active:
            self.model.Proto().ClearField("objective")

        if self._has_a_solution():
            self._start_from_optimum()

        liste_priorites = self.model.objective().get_list_priority()
        list_partial_obj_values = []
        tab_val_par_obj = {}
        list_index_obj_values_constraints = []

        for priority in liste_priorites:
            if not mute:
                logger.info("On maximise au rang {}... ".format(priority))
            must_be_absolute = self.model.objective().must_reach_absolute_max(priority)
            a_maximiser = 0
            for obj in self.model.objective().dic_elt[priority]:
                if isinstance(obj, BonusConstraint):
                    for elementary_constraint in obj.list_constraints():
                        _, boolvar = elementary_constraint
                        a_maximiser = a_maximiser + (obj.coef * boolvar)
                elif isinstance(obj, MinObjective):
                    a_maximiser = a_maximiser - obj.expression()
                elif isinstance(obj, MaxObjective):
                    a_maximiser = a_maximiser + obj.expression()

            self.model.Maximize(a_maximiser)
            status = super().Solve(self.model)

            # We failed to reach the right solution, so we return the error message
            if must_be_absolute:
                if status != cp_model.OPTIMAL:
                    # We release partial optimization values for future explanation questions
                    for i in sorted(list_partial_obj_values, reverse=True):
                        self.model.Proto().constraints[i].Clear()
                    return status, None, None
            else:
                if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
                    for i in sorted(list_partial_obj_values, reverse=True):
                        self.model.Proto().constraints[i].Clear()
                    return status, None, None

            # We fix this objective value as a new constraint
            obj_value = self.Value(a_maximiser)
            list_partial_obj_values.append(obj_value)
            list_index_obj_values_constraints.append(len(self.model.Proto().constraints))
            self.model.Add(a_maximiser >= obj_value)

            # We register the values that we have reached for all pieces of the objective
            for obj in self.model.objective().dic_elt[priority]:
                if isinstance(obj, BonusConstraint):
                    nb_respectee = 0
                    for el_cst in obj.list_constraints():
                        nb_respectee += self.Value(el_cst[1])
                    tab_val_par_obj[obj] = nb_respectee
                else:
                    tab_val_par_obj[obj] = self.Value(obj.expression())

            if not mute:
                logger.info("Score obtenu au rang {} : {}".format(priority, obj_value))
                for obj in self.model.objective().dic_elt[priority]:
                    logger.info("-- {} vaut {}".format(obj.get_id(), tab_val_par_obj[obj]))

            # We remember the current solution as a hint
            self.model.Proto().solution_hint.Clear()
            for v in self.model.dict_indexing().get_list_variables():
                value = self.Value(v)
                self.model.AddHint(v, value)

        status = super().Solve(self.model)

        # We release the constraints that we must reach partial objectives, for future explanation problems
        for i in sorted(list_index_obj_values_constraints, reverse=True):
            self.model.Proto().constraints[i].Clear()

        return status, list_partial_obj_values, tab_val_par_obj

    def Solve(self, model=None, solution_callback=None) -> int:
        if model:
            self.model = model
        if self.model.has_obvious_conflicts():
            self._current_status = Status.OBVIOUS_CONFLICT
            return Status.OBVIOUS_CONFLICT
        status, list_res, tab_val_obj = self.try_solving()
        self._current_status = status
        if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
            self._register_solution(status, list_res, tab_val_obj)
        return status

    def SolveWithSolutionCallback(self, model=None, callback=None) -> int:
        return self.Solve(model, callback)

    def _start_from_optimum(self):
        """
        Takes as hint the solution stored in example_best_solution
        """
        self.model.Proto().solution_hint.Clear()
        for var in self.example_best_solution:
            self.model.AddHint(var, self.example_best_solution[var])

    def _register_solution(self, status, list_part_obj_values, tab_val_obj):
        """
        Keeps in memory the solution passed as argument
        """
        self._current_status = status
        for var in self.model.VarList():
            self.example_best_solution[var] = self.Value(var)
        if list_part_obj_values:
            self.res_optimization = list_part_obj_values
        if tab_val_obj:
            for obj in tab_val_obj:
                obj.set_opt_value(tab_val_obj[obj])

    def _is_better_solution(self, status, list_part_obj_values) -> int:
        """
        Checks if the solution given in argument is an improvement to the one currently in memory
        If not, returns -1
        Else returns the rank of the improvement
        """
        if status != Status.OPTIMAL and status != Status.FEASIBLE:
            return -1
        if not self.res_optimization:
            return 0
        long_sol_actuelle = len(self.res_optimization)
        for i in range(len(list_part_obj_values)):
            if i >= long_sol_actuelle:
                return i
            if self.res_optimization[i] > list_part_obj_values[i]:
                return -1
            if self.res_optimization[i] < list_part_obj_values[i]:
                return i
        return -1

    def LNS(self, lns_variables_choice: LNS_Variables_Choice, max_time_lns):
        """
        Enables you to optimize part of the problem locally

        **See top of the file for examples and additional information**
        """
        tps_max_par_iteration = None
        if max_time_lns:
            tps_max_par_iteration = int(round(max_time_lns / lns_variables_choice.nb_remaining_iterations()))
        while lns_variables_choice.nb_remaining_iterations() > 0:
            next_variables_to_bloc = lns_variables_choice.variables_to_block()
            self._block_part_of_pb_and_optimise(next_variables_to_bloc, tps_max_par_iteration)

    def _block_part_of_pb_and_optimise(self, list_var, max_time_allowed=None) -> None:
        """
        **For internal use only. Use LNS instead**

        Sets list_var to their current value and tries to optimize again
        Assumes that the solver has already been called and has been ran once (otherwise it runs it here)
        """
        ancien_temps = self.parameters.max_time_in_seconds
        if max_time_allowed:
            self.parameters.max_time_in_seconds = max_time_allowed

        # If we have not yen run the problem we launch it to have a starting point
        if self._current_status == Status.NEVER_LAUNCHED:
            logger.info("Solver has not yet been launched. We launch it here on the whole problem")
            self.Solve()

        if not self._has_a_solution():
            logger.info("Problem is infeasible")
            return

        # On bloque temporairement toutes les variables de notre liste (directement dans le Proto)
        list_anciens_domaines = dict()
        for var in list_var:
            idx = self.model.dict_indexing().get_variable_idx(var)
            if idx >= 0:
                current_val = self.Value(var)
                default_range = self.model.Proto().variables[idx].domain[0], self.model.Proto().variables[idx].domain[1]
                list_anciens_domaines[idx] = default_range
                nouv_val = current_val
                self.model.Proto().variables[idx].domain[:] = [nouv_val, nouv_val]
            else:
                logger.info("{} is not a vriable of our problem".format(var))

        # On résout à nouveau avec cette nouvelle config
        status, list_part_obj_values, tab_val_par_obj = self.try_solving(mute=True)

        # Si on a progressé on retient cette solution
        rank_better_sol = self._is_better_solution(status, list_part_obj_values)
        if rank_better_sol >= 0:
            logger.info("Solution was improved at rank {}!".format(rank_better_sol + 1))
            self._register_solution(status, list_part_obj_values, tab_val_par_obj)

        # On se remet dans la position initiale
        for idx in list_anciens_domaines:
            val_bas = list_anciens_domaines[idx][0]
            val_haut = list_anciens_domaines[idx][1]
            self.model.Proto().variables[idx].domain[:] = [val_bas, val_haut]
        self.parameters.max_time_in_seconds = ancien_temps
