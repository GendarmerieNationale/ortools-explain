from abc import abstractmethod, ABC
from typing import List, Tuple

from .constraints import ConstantConstraint


class ObviousConflict(ABC):
    DETECTED = 1
    TWO_CONSTANT_CONFLICT = 2
    ONE_CONSTANT_CONFLICT = 3
    BACKGROUND_CONSTANT_CONFLICT = 4

    @abstractmethod
    def write_conflict(self, model) -> List[str]:
        pass

    @abstractmethod
    def get(self) -> Tuple:
        pass


class ObviousConflictRaiseError(ObviousConflict):
    def __init__(self, expression, sentence):
        self.expression = expression
        self.sentence = sentence

    def write_conflict(self, model):
        return [self.sentence]

    def get(self):
        return ObviousConflict.DETECTED, self.expression

    def __eq__(self, b):
        if isinstance(b, ObviousConflictRaiseError):
            return self.expression == b.expression and self.sentence == b.sentence
        return NotImplemented


class ObviousConflictConstant(ObviousConflict):
    def __init__(self, c1: ConstantConstraint, c2: ConstantConstraint = None):
        self.c1 = c1
        self.c2 = c2

    @staticmethod
    def elegant_name(model, cc: ConstantConstraint):
        mes_dim = cc.dimensions()
        ma_clef = '-'
        for d in mes_dim:
            ma_clef = ma_clef + d + '-'

        nom_macroblock = cc.name_macroblock()
        my_sentence = None
        if model.dict_indexing().has_mb(nom_macroblock):
            my_sentence = model.dict_indexing().get_mb(nom_macroblock).get_explanation_for_key(ma_clef)

        if not my_sentence:
            name = nom_macroblock
            if len(mes_dim) > 0:
                name_dim = " ("
                for d in mes_dim:
                    name_dim = name_dim + d + "=" + str(cc.dim_value(d)) + ", "
                name_dim = name_dim[:-2] + ")"
                name = name + name_dim
            return name

        for d in mes_dim:
            dim_key = '{' + d + '}'
            my_sentence = my_sentence.replace(dim_key, str(cc.dim_value(d)))

        return my_sentence

    def write_conflict(self, model):
        if self.c2:
            return [self.elegant_name(model, self.c1), self.elegant_name(model, self.c2)]
        else:
            return [self.elegant_name(model, self.c1)]

    def get(self):
        if self.c2:
            return ObviousConflict.TWO_CONSTANT_CONFLICT, (self.c1, self.c2)
        else:
            return ObviousConflict.ONE_CONSTANT_CONFLICT, self.c1


class ObviousConflictBackground(ObviousConflict):
    def __init__(self, variable, val1, val2):
        self.v = variable
        self.val1 = val1
        self.val2 = val2

    def write_conflict(self, model):
        return ['La constante {} est définie deux fois dans les background blocks, une fois à {} et une fois à {}'.format(self.v, self.val1, self.val2)]

    def get(self):
        return ObviousConflict.BACKGROUND_CONSTANT_CONFLICT, (self.v, self.val1, self.val2)
