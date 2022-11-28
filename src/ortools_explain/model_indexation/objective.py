"""
Those classes enable us to represent a problem with multiple objectives
Partial objectives are either maximised or minimised sequentially or combined with coefficients
Objectives are looked at by increasing order of rank

"""

import logging

logger = logging.getLogger(__name__)


class PartObjective:
    """
    Part of the objectives of the model
    """

    def __init__(self, idx: str, priority: int):
        """
        :param idx: unique (at this stage) identifier of the partial objective
        :param priority: rank at which this objective will be looked at
        """
        self._priority = priority
        self._best_value = None
        self._idx = idx

    def get_id(self) -> str:
        return self._idx

    def priority(self) -> int:
        return self._priority

    def best_value(self) -> int:
        return self._best_value

    def set_opt_value(self, val) -> None:
        self._best_value = val


class BonusConstraint(PartObjective):
    """
    Constraint that is not mandatory but grants a bonus if respected
    """
    def __init__(self, model, constraint, idx: str, coef: int, priority: int):
        super().__init__(idx, priority)
        self.size = 1
        self.coef = coef
        boolvar = model.NewBoolVar('is_respected {} - {}'.format(idx, self.size))
        model.Add(constraint).OnlyEnforceIf(boolvar)
        self._list_constraints = [(constraint, boolvar)]

    def add(self, model, constraint):
        self.size = self.size + 1
        boolvar = model.NewBoolVar('is_respected {} - {}'.format(self._idx, self.size))
        model.Add(constraint).OnlyEnforceIf(boolvar)
        self._list_constraints.append((constraint, boolvar))

    def list_constraints(self):
        return self._list_constraints

    def size(self):
        return self.size


class MinObjective(PartObjective):
    """
    Part of an objective that we want to minimize
    """

    def __init__(self, expression, idx: str, priorite: int):
        """
        param expression: Linear expression that we want to MINIMIZE (eg. sum(Xij))
        param priorite: Rank of this objective
        """
        super().__init__(idx, priorite)
        self._expression = expression

    def expression(self):
        return self._expression


class MaxObjective(PartObjective):
    """
    Part of an objective that we want to maximize
    """

    def __init__(self, expression, idx: str, priority: int):
        """
        :param expression: Linear expression that we want to MAXIMIZE (eg. sum(Xij))

        :param idx: unique identifier of this partial objective
        """
        super().__init__(idx, priority)
        self._expression = expression

    def expression(self):
        return self._expression


class Objective:
    """
    The Objective class enables us to keep in memory the different parts of optimisation
    """
    def __init__(self):
        self.dic_elt = dict()
        self.dic_ids = dict()
        self._reach_absolute_max = dict()

    def get_id(self, idx):
        if idx in self.dic_ids:
            return self.dic_ids[idx]
        return None

    def add_part_objective(self, idx, objective: PartObjective, max_must_be_absolute: bool):

        priorite = objective.priority()
        if priorite not in self.dic_elt:
            self.dic_elt[priorite] = [objective]
        else:
            self.dic_elt[priorite].append(objective)

        if max_must_be_absolute:
            self._reach_absolute_max[priorite] = True

        self.dic_ids[idx] = objective

    def get_list_priority(self):
        """Returns the list of priorities in decreasing order"""
        return sorted(list(self.dic_elt.keys()))

    def must_reach_absolute_max(self, priority):
        """Returns whether or not we must go to the end of the optimization for this specific step"""
        return priority in self._reach_absolute_max

    def is_empty(self):
        return len(self.dic_elt) == 0

    def get_all_obj(self):
        return list(self.dic_ids.values())

    def best_value_at_rank(self, rank):
        """Computes the current optimisation value at a given rank"""
        if rank not in self.dic_elt:
            return None
        score = 0
        for obj in self.dic_elt[rank]:
            if isinstance(obj, BonusConstraint):
                score = score + (obj.coef * obj.best_value())
            elif isinstance(obj, MinObjective):
                score = score - obj.best_value()
            elif isinstance(obj, MaxObjective):
                score = score + obj.best_value()
        return score
