"""
Abstract class that must be used to associate different kinds of metrics to a specific model.
"""
from typing import *
from abc import ABC, abstractmethod

from ortools.sat.python import cp_model


class Metric(ABC):
    """
    Abstract class for Metric.

    **Example**

    ```
    import numpy as np

    class MetricVariance(Metric):
        def __init__(self, persons, tasks):
            self.persons = persons
            self.tasks = tasks
            self.id = "variance"

        def sentence_producer(self, value):
            return "The variance for number of task is : %d" % value

        def run(self, variable_assigment):
            nb_of_task_by_person = [sum(variable_assigment[i, j] for j in self.tasks) for i in self.persons]
            value = np.var(nb_of_task_by_person)
            return {
                "id_metrics" : self.id,
                "value" : value,
                "sentence" : self.sentence_producer(value)
            }
    ```
    """
    @abstractmethod
    def run(self, variable_assigment: Dict[cp_model.IntVar, int]) -> Dict:
        """
        Returns a dict
        ```
        {
            "id" : str - id of the metric
            "value" : int - value of the metrics
            "sentence": str - setence to print
        )
        ```
        """
        pass


class BasicMetric(Metric):
    """
    Class that enables an easier generation of a Metric, if it is rather simple

    **Example**
    ```
    m = BasicMetric(idx= "plain_sum",
        value_function= lambda dic_values : sum(dic_values[k] for k in dic_values),
        str_function= lambda value : "The sum of all variables is {}".format(value))
    ```

    """
    def __init__(self, idx: str, function_for_value: callable, function_for_sentence: callable):
        self.idx = idx
        self.f_sentence = function_for_sentence
        self.f_value = function_for_value

    def run(self, variable_assignment: Dict[cp_model.IntVar, int]) -> Dict:
        return {"id": self.idx,
                "value": self.f_value(variable_assignment),
                "sentence": self.f_sentence(variable_assignment)
                }
