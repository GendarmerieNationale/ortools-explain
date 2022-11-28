"""Class used as enum to return the current status of the solver (*SuperSolver* object)."""


class Status:
    """Class used as enum to return the current status of the solver (*SuperSolver* object).
    At any given time you can get the current status of the solver by calling *status()* on it.

    Possible status are:

    * UNKNOWN -- similar to cp_model.UNKOWN
    * MODEL_INVALID -- similar to cp_model.MODEL_INVALID
    * FEASIBLE -- similar to cp_model.FEASIBLE
    * INFEASIBLE -- similar to cp_model.INFEASIBLE
    * OPTIMAL -- similar to cp_model.OPTIMAL
    * OBVIOUS_CONFLICT -- the module has not proceeded to launching the solver because an obvious conflict has already been
    detected in the model (for instance a variable has been set to a constant twice with different values)
    * NEVER_LAUNCHED -- the solver has not yet been launched (call *Solve()*)

    """
    UNKNOWN = 0
    MODEL_INVALID = 1
    FEASIBLE = 2
    INFEASIBLE = 3
    OPTIMAL = 4
    OBVIOUS_CONFLICT = 5
    NEVER_LAUNCHED = 6
