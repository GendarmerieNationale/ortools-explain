import logging
from ortools.sat.python import cp_model

logger = logging.getLogger(__name__)


def is_consistent(model, solver):
    """
    Function that return True if the problem is solvable or else False.
    :return: (boolean)
    """
    status = solver.Solve(model)
    if status == cp_model.INFEASIBLE:
        return False
    elif status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        return True
    else:
        logger.error("Status UNKNOWN at the end of the resolution")
        raise ValueError("Status UNKNOWN at the end of the resolution")
