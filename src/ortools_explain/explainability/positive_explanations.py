import logging
from typing import Dict

from ortools_explain.status import Status

logger = logging.getLogger(__name__)


def positive_explanations(solver, additional_condition) -> Dict:
    """Local explicability. The solver computes new objective values with an additional condition

    Returns one of the following:
    ```
    res = positive_explanations(my_solver, my_condition)

    # Adding this condition makes the model infeasible
    >> res = {"outcome": "infeasible"}

    # OR adding this condition lowers the optimization value at some point
    >> res = {"outcome": "less_optimal",
        "optimization_scores": {"old_values": [v1, v2...],
                                "new_values": [v'1, v'2...]},
        "objective_values": [{"id": obj1, "old_value": v1, "new_value": v'1}... ]
        }

    # OR another solution can be found with this additional condition without lowering the objective values
    >> res = {"outcome": "as_optimal",
        "changed_variables": [{"name": x_1_1, "old_value": v1, "new_value": v'1]}... }
    ```

    """
    model = solver.model
    # On ajoute comme nouvelle contrainte que variable ne doit pas prendre la valeur actuelle
    ref_contrainte = len(model.Proto().constraints)
    model.Add(additional_condition)

    # On résout avec cette nouvelle contrainte
    status, list_val, tab_val_obj = solver.try_solving(mute=True)

    if status != Status.FEASIBLE and status != Status.OPTIMAL:
        res = {"outcome": "infeasible"}

    elif list_val == solver.res_optimization:
        res = {"outcome": "as_optimal", "changed_variables": []}
        for var in model.VarList():
            if "is_respected" not in var.Name():
                if solver.Value(var) != solver.example_best_solution[var]:
                    res["changed_variables"].append({"name": var.Name(),
                                                     "old_value": solver.example_best_solution[var],
                                                     "new_value": solver.Value(var)})

    else:
        res = {"outcome": "less_optimal",
               "optimization_scores": {"old_value": solver.res_optimization,
                                       "new_value": list_val},
               "objective_values": []
               }

        for obj in model.objective().get_all_obj():
            res["objective_values"].append({"id": obj.get_id(), "old_value": obj.best_value(), "new_value": tab_val_obj[obj]})

    # On supprime cette nouvelle contrainte
    model.Proto().constraints[ref_contrainte].Clear()

    return res
