# Installation

This module is made to be an overlayer of the Google OR Tools cpsolver.
It does not require any python library other than **ortools**.

Full documention is available at : https://datalab-stsisi.github.io/

# Overview

The module enables you to perform in an easy way some operations on cp models:

* Modelization of multiple objectives in a single model, with sequential or combined optimization
* Modelization of relaxable constraints (constraints that are not mandatory but award bonuses if respected)
* Explanation of infeasibility for infeasible problems
* Local explanation of solution for feasible problems
* Production of natural language explanations
* Easier local optimization (LNS)

The **examples** package provides two files with simple examples of use of the module functions on sudoku examples.

## SuperModel

The **SuperModel** class is an overlayer of ortools cp_model which enables us to name constraints (useful for negative explainability) and
define multiple objectives, including ones consisting of allowing or not the relaxation of a constraint.

### Negative explainability

#### *Add* and *AddConstant*

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
model.Add(X[i, j1] + X[i, j2] == 1, "uniqueness_of_job", person='person with id %d'%i)
```

Constants are specific kinds of constraints. In the basic OR Tools module, constants are defined
with *model.NewConstant(value)*.

In our module, constants are defined in a way that is in most aspects similar to the other constraints, with the *AddConstant* method.
Constants that are defined with *AddConstant* must always be declared as variables beforehand:

```
X[i, j, k] = model.NewBoolVar('assignment_{}_{}_{}'.format(i, j, k))
model.AddConstant(X[i, j, k], 0, "initial_assignment", person=i, mission=j)
```

#### Correct and incorrect implementations

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

# INCORRECT - the same type cannot be given to a usual type of constraint (declared with Add) and a constant one (declared with AddConstant)
model.Add(a == b, "my_type")
model.AddConstant(var, value, "my_type")
```

#### *AddExplanation*

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
"{person} can have no more than one job on day {day}")
```

The explanations produced will now be:
```
>> "A person can have no more than one job at any given day"
>> "p1 can have no more than one job at any given day"
>> "People can have no more than one job on day d2"
>> "p1 can have no more than one job on day d2"
```

## Multi-objective modelization

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

### Correct and Incorrect implementations

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

## SuperSolver

This class is an overlayer to OR Tools's solver. It allows for multi-optimization solving and local positive explainability.

### Solving with SuperSolver

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

#### Example
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

  ### Local optimization

SuperSolver contains a LNS function that enables you to improve the solution found by the solver by searching for local improvements.

To do this, you decide to set part of the problem to its current value and only allow changes in the rest of the problem.

To use LNS techniques, you must create a class that inherits from the *LNS_Variables_Choice* abstract class.
Classes inheriting from *LNS_Variables_Choice* need two methods:

* **variables_to_block()** must return at each call the variables that will not be allowed to change from their current value
* **nb_remaining_iterations()** must return at each call the number of times to optimize again locally

#### Example - Scheduling problem

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

  ```
  if my_solver.status() == Status.FEASIBLE:
      # We create a dictionary that links each variable to its value on the dimension of interest (here days)
      dict_for_lns = {X[i, j, k]: j for (i, j, k) in product(list_people, list_days, list_missions)}
      # We simply define the LNS strategy with the built-in function
      lns_strategy = LNS_Variables_Choice_Across_One_Dim(dict_for_lns, window_size= 5, step= 2, nb_iterations= 1)

      my_solver.LNS(lns_strategy, max_time_lns= 300)
  ```

  ### Local explainability

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
