"""
Defines a SuperModel class that overrides cp_model and enables you to name constraints, add relaxable constraints, define
multiple objectives...
"""
import logging

from typing import *

from ortools.sat.python.cp_model import BoundedLinearExpression, IntVar
from ortools.sat.python import cp_model

from ortools_explain.model_indexation.constraints import (
    DictIndexConstraints,
    ConstantConstraint,
    ModelisationError,
    UsualConstraint,
)
from ortools_explain.model_indexation.obvious_conflict import ObviousConflictConstant, ObviousConflictBackground, ObviousConflict, ObviousConflictRaiseError
from ortools_explain.model_indexation.objective import Objective, MinObjective, MaxObjective, BonusConstraint

from .metric import Metric

logger = logging.getLogger(__name__)


class SuperModel(cp_model.CpModel):
    """

    # Summary

    This class is an overlayer of ortools cp_model which enables us to name constraints (useful for negative explainability) and
    define multiple objectives, including ones consisting of allowing or not the relaxation of a constraint.

    # Negative explainability

    ## *Add* and *AddConstant*

    Functions *Add* and *AddConstant* are called while constructing the model.
    They will give the module an idea of the structure of the problem so that the module can return which parts of the problem
    are in conflict if the problem is infeasible.

    Constraints are declared in the following fashion:

    ```
    model.Add(X[i, j1] + X[i, j2] == 1, "uniqueness_of_job", person=i)
    ```

    where "uniqueness_of_job" is the **general type** of the constraint, and "person=i" is the only **dimension**, which key is "person"
    and which value is i. Constraints can have 0, 1 or several dimensions.

    All the examples below are correct calls of this function:

    ```
    # It is not mandatory to give type and dimensions. Such constraints are
    # called "background blocks", and can never be released
    model.Add(X[i, j1] + X[i, j2] == 1)

    # Dimensions are not mandatory
    model.Add(X[i, j1] + X[i, j2] == 1, "uniqueness_of_job")

    model.Add(X[i, j1] + X[i, j2] == 1, "uniqueness_of_job", person=i)
    model.Add(X[i, j1] + X[i, j2] == 1, "uniqueness_of_job", person=i, first_day=j1)
    model.Add(X[i, j1] + X[i, j2] == 1, "uniqueness_of_job", person=i, first_day=j1, second_day=j2)
    model.Add(X[i, j1] + X[i, j2] == 1, "uniqueness_of_job", person=i, days=('{}-{}'.format(j1, j2)))
    model.Add(X[i, j1] + X[i, j2] == 1, "uniqueness_of_job", person=i+1)
    model.Add(X[i, j1] + X[i, j2] == 1, "uniqueness_of_job", person='person with id %d'%1)
    ```

    Constants are specific kinds of constraints. In the basic OR Tools module, constants are defined
    with *model.NewConstant(value)*.

    In our module, constants are defined in a way that is in most aspects similar to the other constraints, with the *AddConstant* method.
    Constants that are defined with AddConstant must always be declared as variables beforehand:

    ```
    X[i, j, k] = model.NewBoolVar('assignment_{}_{}_{}'.format(i, j, k))
    model.AddConstant(X[i, j, k], 0, "initial_assignment", person=i, mission=j)
    ```

    ## Correct and incorrect implementations

    ```
    # --- CONSTANTS ---

    # CORRECT - constants can also be defined as background blocks (without general type and dimensions)
    model.AddConstant(x[i, j], 0)

    # BAD PRACTICE - the implementation below will work but may lead to less precise explanations for infeasible problems
    # Use implementation above instead
    x[i, j] = model.NewConstant(0)

    # --- DEFINITION OF DIMENSIONS ---

    # INCORRECT - it is incorrect to give dimensions without a general type
    model.Add(X[i, j1] + X[i, j2] == 1, person=i)

    # INCORRECT - dimension values cannot contain '{' or '}'
    model.Add(X[i, j1] + X[i, j2] == 1, "my_type", my_dimension = "}stupid_value{")

    # INCORRECT - dimension values must be of hashable type (eg. str, int) and therefore cannot be modifiable objects such as list or dict
    model.Add(X[i, j1] + X[i, j2] == 1, "my_type", my_dimension = "[j1, j2]")

    # CORRECT - dimension values can be anything hashable
    model.Add(X[i, j1] + X[i, j2] == 1, "my_type", my_dimension = "(j1, j2)")

    # --- COMBINATION OF CONSTRAINTS ---

    # CORRECT - several constraints can be given the same type and the same dimensions
    model.Add(a + b > 0, "sum", id=a)
    model.Add(a + c > 0, "sum", id=a)

    # CORRECT - a variable can be assigned as a constant twice with different values.
    # (This type of problem is obviously infeasible but the modelization itself is correct)
    model.AddConstant(pos[0, 0, 1], 1, "position_to_1")
    model.AddConstant(pos[0, 0, 1], 0, "position_to_0")

    # CORRECT - several constants can be declared with the same general type and the same dimensions even if the value is not the same
    model.AddConstant(pos[0, 0, 1], 1, "position", x=0, y=0)
    model.AddConstant(pos[0, 0, 2], 0, "position", x=0, y=0)

    # INCORRECT - constraints with the same general type must have homogeneous dimensions (same keys)
    model.Add(a == b, "my_type")
    model.Add(c == d, "my_type", dim1=c)

    # INCORRECT - the same type cannot be given to a usual type of constraint (declared with *Add*) and a constant one (declared with *AddConstant*)
    model.Add(a == b, "my_type")
    model.AddConstant(var, value, "my_type")
    ```

    ## *AddExplanation*

    AddExplanation enables you to link a constraint to a natural language explanation that a user can understand.

    Considering the set of constraints:
    ```
    for i in list_persons:
        for j in list_days:
            model.Add(X[i, j, k1] + X[i, j, k2] <= 1, "no_more_than_one_job", person=i, day=j)
    ```

    Depending on the situation, the solver for infeasibility may for instance return one of the following:
    ```
    >> "no_more_than_one_job"
    >> "no_more_than_one_job" (person= p1)
    >> "no_more_than_one_job" (day= d2)
    >> "no_more_than_one_job" (person= p1, day= d2)
    ```

    Which may not be easy to understand.

    By adding the following lines:
    ```
    model.AddExplanation("no_more_than_one_job",
    "A person can have no more than one job at any given day",
    "{person} can have no more than one job at any given day",
    "People can have no more than one job on day {day}",
    "{person} can have no more than one job on day {day})
    ```

    The explanations produced will now be:
    ```
    >> "A person can have no more than one job at any given day"
    >> "p1 can have no more than one job at any given day"
    >> "People can have no more than one job on day d2"
    >> "p1 can have no more than one job on day d2"
    ```

    # Multi-objective modelization

    The module enables you to implement several objectives in an optimization problem, where OR Tools only allows for one.

    Partial objectives are defined with the following functions.

    ```
    # Adds a constraint that is not mandatory, but will increase the optimization value (here by 50 points) if respected
    model.AddRelaxableConstraint(assignment[i_1, j] + assignment[i_2, j] == 1, idx= "ops_need", coef= 50, priority= 1)

    # Adds an objective that must be maximized
    model.AddMaximumObjective(goal_1, idx= "first goal", priority= 2)

    # Adds an objective that must be minimized
    model.AddMinimumObjective(2 * goal_2, idx= "second goal", priority= 2)
    ```

    The module will process optimization by increasing rank of priority. Several objectives can be defined for the same priority,
    in which case they will be combined.

    ## Correct and Incorrect implementations

    ```
    # INCORRECT - coef must be a positive integer
    model.AddRelaxableConstraint(x[i_1, j] + x[i_2, j] == 2, idx= "punish combination", coef= -50, priority= 1)
    # POSSIBLE CORRECTION:
    model.AddRelaxableConstraint(x[i_1, j] + x[i_2, j] != 2, idx= "reward no combination", coef= 50, priority= 1)

    # ALL INCORRECT - priority must be a positive integer
    model.AddMaximumObjective(my_sum, idx= "combination", priority= 0)
    model.AddMaximumObjective(my_sum, idx= "combination", priority= 0.5)
    model.AddMaximumObjective(my_sum, idx= "combination", priority= -1)

    # INCORRECT - Idx must all be distinct with one exception (see next example)
    model.AddRelaxableConstraint(x[0, 1] + x[0, 2] == 2, idx= "combination", coef= 50, priority= 1)
    model.AddMaximumObjective(my_sum, idx= "combination", priority= 1)

    # CORRECT - Two relaxable constraints can be given the same idx, only if they have same coef and priority
    model.AddRelaxableConstraint(x[0, 1] + x[0, 2] == 2, idx= "combination", coef= 50, priority= 1)
    model.AddRelaxableConstraint(x[1, 1] + x[1, 2] == 2, idx= "combination", coef= 50, priority= 1)

    # INCORRECT - Same idx and different priorities
    model.AddRelaxableConstraint(x[0, 1] + x[0, 2] == 2, idx= "combination", coef= 50, priority= 1)
    model.AddRelaxableConstraint(x[1, 1] + x[1, 2] == 2, idx= "combination", coef= 50, priority= 2)

    # INCORRECT - Same idx and different coef
    model.AddRelaxableConstraint(x[0, 1] + x[0, 2] == 2, idx= "combination", coef= 50, priority= 1)
    model.AddRelaxableConstraint(x[1, 1] + x[1, 2] == 2, idx= "combination", coef= 80, priority= 1)
    ```

    """

    def __init__(self):
        super().__init__()
        self._dict_indexing = DictIndexConstraints()
        self._objective = Objective()
        self._dico_bool_variables = dict()
        self._dico_int_variables = dict()
        self._inner_conflicts = []
        self._my_metrics = []

    def has_obvious_conflicts(self) -> bool:
        """
        An *obvious conflict* is a conflict that can be detected during the generation of the model,
        or just before solving with the *RaiseErrorIf* function.

        *Example:*

        If both the lines below are met while constructing the model, the module will detect the conflict and store it internally.
        ```
        model.AddConstant(x[i,j], 1)
        model.AddConstant(x[i,j], 0)
        ```
        """
        return len(self._inner_conflicts) > 0

    def list_obvious_conflicts(self) -> List[ObviousConflict]:
        """
        Returns the list of obvious conflicts that have been found at this stage
        """
        return self._inner_conflicts

    def objective(self) -> Objective:
        """
        ** For internal use only **

        Returns an *Objective* object, which is used for multi-objective optimization and also for local positive explainability
        """
        return self._objective

    def dict_indexing(self) -> DictIndexConstraints:
        """
        ** For internal use only **

        Returns the DictIndexConstraints object which is used for negative explainability
        """
        return self._dict_indexing

    def get_variable(self, id_var) -> Union[IntVar, None]:
        """
        ** For internal use only **

        Returns the variable at given position in the Proto, for instance to define it as a constant
        """
        if id_var in self._dico_bool_variables:
            return self._dico_bool_variables[id_var]
        if id_var in self._dico_int_variables:
            return self._dico_int_variables[id_var]
        return None

    def VarList(self) -> List[IntVar]:
        """
        ** For internal use only **

        Returns the list of variables currently in the model
        """
        return list(self._dico_bool_variables.values()) + list(self._dico_int_variables.values())

    def NewBoolVar(self, name):
        """
        Overlay of cp_model.NewBoolVar that enables us to index the variable
        """
        x = super().NewBoolVar(name)
        self.dict_indexing().add_variable(x, len(self.Proto().variables) - 1)
        self._dico_bool_variables[name] = x
        return x

    def NewIntVar(self, lb, ub, name):
        """
        Overlay of cp_model.NewIntVar that enables us to index the variable
        """
        x = super().NewIntVar(lb, ub, name)
        self.dict_indexing().add_variable(x, len(self.Proto().variables) - 1)
        self._dico_int_variables[name] = x
        return x

    def Add(self, ct: BoundedLinearExpression, name: str = None, **dimensions):
        """
        Overlay of cp_model.Add that enables you to link a constraint with a name and a dimension.

        **name**: the "general type" of this constraint

        **dict_dimensions**: the dimensions of the constraint block (dictionary containing both keys and values)

        **See top of the file for examples and additional information**
        """
        if dimensions and not name:
            raise ModelisationError("You cannot assign dimensions to a constraint without a general type")

        if name:
            self._check_valid_dim(**dimensions)

            my_idx = len(self.Proto().constraints)
            block = UsualConstraint(name, my_idx, **dimensions)
            self._dict_indexing.add_usual_constraint(block, my_idx)
        return super().Add(ct)

    def AddConstant(self, variable: IntVar, default_value: int, name=None, **dimensions) -> None:
        """
        Method that replaces *model.NewConstant()* and enables you to declare a constant with a name and dimension

        **variable**: variable that will be set to a constant value

        **default_value**: this value

        **name**: the "general type" of this constraint

        **dict_dimensions**: the dimensions of the constraint block (dictionary containing both keys and values)

        **See top of the file for examples and additional information**
        """
        if dimensions and not name:
            raise ModelisationError("You cannot assign dimensions to a constraint without a general type")

        if name:
            self._check_valid_dim(**dimensions)

        idx = self.dict_indexing().get_variable_idx(variable)
        if idx == -1:
            raise ModelisationError("Constant {} must be declared as a variable before it is declared as a constant".format(variable))

        val_bas = self.Proto().variables[idx].domain[0]
        val_haut = self.Proto().variables[idx].domain[1]

        # We have already set the same variable to a constant value
        if val_bas == val_haut:
            val = val_bas

            # We have defined it twice with the same value - no problem but no more work to do
            if val == default_value:
                logger.debug("Variable {} has already been set to the constant value {}".format(variable, val))
                return
            # We have defined it twice with different values - this is an obvious conflict, we record it
            else:
                ancienne_cc = None
                if self.dict_indexing().has_constant_block(idx):
                    ancienne_cc = self.dict_indexing().get_constant_block(idx)

                if ancienne_cc and name:
                    cc = ConstantConstraint(name, idx, default_value, (0, 0), **dimensions)
                    self._inner_conflicts.append(ObviousConflictConstant(ancienne_cc, cc))
                elif ancienne_cc and not name:
                    self._inner_conflicts.append(ObviousConflictConstant(ancienne_cc))
                elif not ancienne_cc and name:
                    cc = ConstantConstraint(name, idx, default_value, (0, 0), **dimensions)
                    self._inner_conflicts.append(ObviousConflictConstant(cc))
                else:
                    self._inner_conflicts.append(ObviousConflictBackground(variable, val, default_value))
                return

        else:
            # On implémente réellement le fait que c'est actuellement une constante
            default_range = (self.Proto().variables[idx].domain[0], self.Proto().variables[idx].domain[1])
            self.Proto().variables[idx].domain[:] = [default_value, default_value]

            if name:
                c = ConstantConstraint(name, idx, default_value, default_range, **dimensions)
                self._dict_indexing.add_constant_constraint(c, idx)

    def AddExplanation(self, gnl_type, *sentences):
        """Enables you to write an explanation in natural language for users to understand

        **gnl_type**: general type of constraints for which these explanations will be produced

        **sentences**: list of explanations

        **See top of the file for examples and additional information**
        """
        if gnl_type in self.dict_indexing().list_macro_block_names():
            self.dict_indexing().get_mb(gnl_type).add_explanations(*sentences)
        else:
            logger.info("The type of constraints \"{}\" does not exist in our model".format(gnl_type))

    def RaiseErrorIf(self, equation, sentence: str) -> None:
        """Enables us to perform basic checks before launching the solver

        **equation**: If this is true, then we believe the model is always infeasible and we will not even proceed to trying to launch it

        **sentence**: Natural language explanation to produce if an error is lifted at this point

        *RaiseErrorIf* should be called after creating the model and before launching the actual Solve.

        *Example:*
        ```
        model.RaiseErrorIf(nb_hours_needed_week > 40 * nb_people_available,
            sentence= "{} hours are needed this week which is not compatible with a 40h-a-week limit".format(nb_hours_needed_week))
        ```
        """
        if equation:
            conflict = ObviousConflictRaiseError(equation, sentence)
            if conflict not in self._inner_conflicts:
                self._inner_conflicts.append(conflict)

    def AddMetric(self, metric: Metric):
        """
        Enables you to define metrics you want to compute with every resolution of the problem
        """
        self._my_metrics.append(metric)

    def Metrics(self) -> List[Metric]:
        """Returns the list of metrics currently stored in the model"""
        return self._my_metrics

    def AddRelaxableConstraint(self, expression, idx: str, coef: int, priority: int, must_be_optimal=False):
        """
        Enables you to define a constraint that is not mandatory but will award points if respected

        **expression**: This constraint

        **idx**: Identifier for this constraint. Idx may not be unique but the same idx
        can only be given to relaxable constraints of same coef and priority

        **coef**: Bonus that respecting this constraint will grant you

        **priority**: Rank at which this constraint will be decided on

        **must_be_optimal**: If False, the module will enable optimization at the next rank even if the absolute
        optimum has not be reached, so as to return a result in reasonable time

        **See top of the file for examples and additional information**
        """
        if coef <= 0:
            raise ModelisationError("Coef must be a strictly positive integer")
        if priority <= 0:
            raise ModelisationError("Priority must be a strictly positive integer")

        already_built = self.objective().get_id(idx)
        if already_built:
            if not isinstance(already_built, BonusConstraint):
                raise ModelisationError("The id {} is given both to a relaxable constraint and to another type of objective".format(idx))
            old_prio = already_built.priority()
            old_coef = already_built.coef
            if old_prio != priority:
                raise ModelisationError("Relaxable constraint {} is defined twice with the same name but with different priorities".format(idx))
            if old_coef != coef:
                raise ModelisationError("Relaxable constraint {} is defined twice with the same name but with different coefficients".format(idx))
            already_built.add(self, expression)
        else:
            self.objective().add_part_objective(idx, BonusConstraint(self, expression, idx, coef, priority), max_must_be_absolute=must_be_optimal)

    def AddMinimumObjective(self, expression, idx: str, priority: int, must_be_optimal=False):
        """Enables you to define a linear expression as one of the objectives of your model

        **expression**: The expression to minimize

        **idx**: Unique identifier of this partial objective

        **priority**: Rank at which this objective will be minimized

        **must_be_optimal**: If False, the module will enable optimization at the next rank even if the absolute
        optimum has not be reached, so as to return a result in reasonable time

        **See top of the file for examples and additional information**
        """
        if priority <= 0:
            raise ModelisationError("Priority must be a strictly positive integer")
        if self.objective().get_id(idx):
            raise ModelisationError("The id {} is given to two different types of objective".format(idx))
        self._objective.add_part_objective(idx, MinObjective(expression, idx, priority), max_must_be_absolute=must_be_optimal)

    def AddMaximumObjective(self, expression, idx: str, priority: int, must_be_optimal=False):
        """Enables you to define a linear expression as one of the objectives of your model

        **expression**: The expression to maximize

        **idx**: Unique identifier of this partial objective

        **priority**: Rank at which this objective will be maximized

        **must_be_optimal**: If False, the module will enable optimization at the next rank even if the absolute
        optimum has not be reached, so as to return a result in reasonable time

        **See top of the file for examples and additional information**
        """
        if priority <= 0:
            raise ModelisationError("Priority must be a strictly positive integer")
        if self.objective().get_id(idx):
            raise ModelisationError("The id {} is given to two different types of objective".format(idx))
        self._objective.add_part_objective(idx, MaxObjective(expression, idx, priority), max_must_be_absolute=must_be_optimal)

    @staticmethod
    def _check_valid_dim(**dimensions):
        for d in dimensions.keys():
            if "{" in d or "}" in d:
                raise ModelisationError("Dimension names cannot contain { or }")
            val = dimensions[d]
            if not val.__hash__:
                raise ModelisationError("{} is not an acceptable dimension value type".format(type(val)))
