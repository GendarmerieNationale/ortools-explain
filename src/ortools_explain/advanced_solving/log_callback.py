import traceback
from abc import ABC, abstractmethod
from time import time
from typing import *
import logging
import json

from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import IntVar

from ortools_explain.model import SuperModel


logger = logging.getLogger(__name__)


class SolutionWriter(ABC):
    """
    Abstract class for a way to write the solution in a user-friendly way
    Run returns a dictionary or a list that describes your solution in a way that is useful to you

    **Example**
    ```

    # SchedulingProblem
    # X[i, j, k] are Boolean variables
    # X[i, j, k] = 1 <=> Person i on day j will work on job k

    class SolutionWriterSchedule(SolutionWriter):
        def get_position(dic_values, i, j):
            for k in list_positions:
                if dic_values[X[i, j, k]] == 1:
                    return k
            return "no position"

        def run(dic_values):
            dico_res = {"schedule": []}
            for i in list_persons:
                planning_pers = {"id_staff": i, "jobs": []}
                for j in list_days:
                    planning_pers["jobs"] = get_position(dic_values, i, j)
            return dico_res

    """
    @abstractmethod
    def run(self, dic_values: Dict[IntVar, int]) -> Dict:
        """
        Returns a dict of any shape
        """
        pass


class BasicWriter(SolutionWriter):
    """
    Simply returns the value of every variable of the problem
    """
    def run(self, dic_values):
        dic = {"values": []}
        for x in dic_values:
            if 'is_respected' not in x.Name():
                dic["values"].append({"id": x.Name(), "value": dic_values[x]})
        return dic


class LogCallback(cp_model.CpSolverSolutionCallback):
    """
    Overlayer of cp_model.CpSolverSolutionCallback

    It enables you to dump all information on the problem in a json at each stage

    Data is stored in this fashion:

    {
        "time": [t1, t2, ...],
        "solution": [solution1, solution2, ...],
        "optimisation_values": {
            "rank_1": [],
            "rank_2": []...
        "objective_values" : {
            "id_obj_1" : [v1, v2, ...],
            "id_obj_2": [...]
        }
        "metric_values: {
            "id_metrics_1" : {
                "value": [v1_m1, v2_m1, ...];
                "sentence" : [s1_m1, s2_m2, ...];
            };
            "id_metrics_2 : {
                ...
            };
        ...
    }
    """
    def __init__(
        self,
        model: SuperModel,
        where_to_log_metric_update="",
        solution_writer=None,
        dump_metric=False,
        min_time_between_update: int = None,
    ):
        super().__init__()
        self.model = model
        self.initial_time = time()
        self.where_to_log_metric_update = where_to_log_metric_update
        self.min_time_between_update = min_time_between_update
        self.solution_writer = solution_writer
        self.dump_metric = dump_metric

    def on_solution_callback(self):
        try:
            X = {var: self.Value(var) for var in self.model.VarList()}

            # On récupère le dernier time
            output_metric = json.load(open(self.where_to_log_metric_update, "r"))

            too_soon = False
            if self.min_time_between_update:
                ma_liste = output_metric.get("time", [])
                if len(ma_liste) > 0:
                    last_time = ma_liste[0]
                    if (time() - last_time) < self.min_time_between_update:
                        too_soon = True

            if not too_soon:
                # Writing the new time
                output_metric.get("time", []).append(time() - self.initial_time)

                # Writing the new solution
                if self.solution_writer:
                    dic_solution = self.solution_writer.run(X)
                    output_metric.get("solution", []).append(dic_solution)  # TODO vérifier que ça fonctionne

                # Writing the metrics
                if self.dump_metric:
                    for metric in self.model.Metrics():
                        result_metric = metric.run(X)

                        output_metric.get("metric_values", {})[result_metric["id"]].get("value", []).append(result_metric["value"])
                        output_metric.get("metric_values", {})[result_metric["id"]].get("sentence", []).append(result_metric["sentence"])
                        logger.info("Processing... " + result_metric["sentence"])

                # Writing the partial objectives
                list_obj = self.model.objective().get_all_obj()
                for obj in list_obj:
                    output_metric.get("objective_values", {})[obj.get_id()].append(obj.best_value())
                    logger.info("Processing... Objective {} is currently {}".format(obj.get_id(), obj.best_value()))

                # Writing the optimisation values
                list_priorities = self.model.objective().get_list_priority()
                list_val = []
                for rank in list_priorities:
                    val = self.model.objective().best_value_at_rank(rank)
                    list_val.append(val)
                    output_metric.get("optimisation_values", {})["rank {}".format(rank)].append(val)
                logger.info("Processing... Optimisation values at ranks {} are currently {}".format(list_priorities, list_val))

                json.dump(output_metric, open(self.where_to_log_metric_update, "w"))

        except Exception as exception:
            traceback.print_exc()
            # Permet de logguer correctement les exceptions qui surviennent dans le CallBack
            raise exception
