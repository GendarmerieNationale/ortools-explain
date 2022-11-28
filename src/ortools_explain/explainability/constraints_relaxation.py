import copy
import logging
from typing import *

from ortools_explain.model_indexation.constraints import NamedBlock, UsualConstraint, ConstantConstraint

logger = logging.getLogger(__name__)


class ConstraintsRelaxation:
    """
    Class used to release or reset a constraint or a block of constraints, depending on the chosen method
    """

    def __init__(self, model):
        self.model = model
        self.dict_indexing = self.model.dict_indexing()
        self.is_relaxed_usual_constraint = {uc: None for uc in self.dict_indexing.list_usual_constraints()}  # UsualConstraint --> None si pas relachée, dico des contraintes du proto sinon
        self.is_relaxed_constant = {cc: False for cc in self.dict_indexing.list_constant_constraints()}  # ConstantConstraint --> Boolean, True ssi la contrainte est relachée

    def relax_all_real_constraints_but(self, all_blocks):
        """Releases all true constraints except for those which are in all_blocks"""
        uc_a_garder = {}
        for nb in all_blocks:
            if nb.is_true_constraint():
                for ec in nb.contenu():
                    uc_a_garder[ec] = True

        for uc in self.is_relaxed_usual_constraint:
            if not self.is_relaxed_usual_constraint[uc] and uc not in uc_a_garder:
                self._relax_single_usual_constraint(uc)

    def relax_all_but(self, all_blocks, method):
        """
        Releases all constraints that are not in all_blocks
        If method == 'sufficient assumption', we do not relax constant constraints
        """
        uc_a_garder = {}
        cc_a_garder = {}
        for nb in all_blocks:
            if nb.is_true_constraint():
                for ec in nb.contenu():
                    uc_a_garder[ec] = True
            elif method != 'sufficient_assumption':
                for ec in nb.contenu():
                    cc_a_garder[ec] = True

        for uc in self.is_relaxed_usual_constraint:
            if not self.is_relaxed_usual_constraint[uc] and uc not in uc_a_garder:
                self._relax_single_usual_constraint(uc)
        if method != 'sufficient_assumption':
            for cc in self.is_relaxed_constant:
                if not self.is_relaxed_constant[cc] and cc not in cc_a_garder:
                    self._relax_single_constant_constraint(cc)

    def _relax_single_usual_constraint(self, uc: UsualConstraint):
        """Relaxes one single constraint (checks that it should be relaxed and then calls core function)"""
        li = uc.list_idx()
        self.is_relaxed_usual_constraint[uc] = {}
        for i in li:
            self.is_relaxed_usual_constraint[uc][i] = copy.copy(self.model.Proto().constraints[i])
            self.model.Proto().constraints[i].Clear()

    def _relax_single_constant_constraint(self, cc: ConstantConstraint):
        """Relaxes one single constraint (checks that it should be relaxed and then calls core function)"""
        li = cc.list_idx()
        for i in li:
            self.is_relaxed_constant[cc] = True
            default_range = cc.def_range(i)
            self.model.Proto().variables[i].domain[:] = [default_range[0], default_range[1]]

    def relax_list_blocks(self, my_blocks: List[NamedBlock]):
        """Function that releases a list of elementary blocks"""
        list_uc = []
        list_cc = []
        for mb in my_blocks:
            if mb.is_true_constraint():
                list_uc = list_uc + mb.contenu()
            else:
                list_cc = list_cc + mb.contenu()

        for uc in list_uc:
            if not self.is_relaxed_usual_constraint[uc]:
                self._relax_single_usual_constraint(uc)
        for cc in list_cc:
            if not self.is_relaxed_constant[cc]:
                self._relax_single_constant_constraint(cc)

    def inv_relax_all(self):
        """Reinforces all constraints"""
        for uc in self.is_relaxed_usual_constraint:
            if self.is_relaxed_usual_constraint[uc]:
                self._inv_relax_single_usual_constraint(uc)

        for cc in self.is_relaxed_constant:
            if self.is_relaxed_constant[cc]:
                self._inv_relax_single_constant_constraint(cc)

        self.is_relaxed_usual_constraint = {uc: None for uc in self.dict_indexing.list_usual_constraints()}  # UsualConstraint --> None si pas relachée, contrainte du proto sinon
        self.is_relaxed_constant = {cc: False for cc in self.dict_indexing.list_constant_constraints()}  # ConstantConstraint --> Boolean, True ssi la contrainte est relachée

    def _inv_relax_single_usual_constraint(self, uc: UsualConstraint):
        li = uc.list_idx()
        for i in li:
            del self.model.Proto().constraints[i]
            self.model.Proto().constraints.insert(i, self.is_relaxed_usual_constraint[uc][i])

    def _inv_relax_single_constant_constraint(self, cc: ConstantConstraint):
        li = cc.list_idx()
        for i in li:
            default_value = cc.def_value(i)
            self.model.Proto().variables[i].domain[:] = [default_value, default_value]

    def inv_relax_list_blocks(self, my_blocks: List[NamedBlock], list_usual_idx_not_to_release=None, list_constant_idx_not_to_release=None):
        """Function that releases a list of elementary blocks"""
        if not list_usual_idx_not_to_release:
            list_usual_idx_not_to_release = {}
        if not list_constant_idx_not_to_release:
            list_constant_idx_not_to_release = {}

        list_uc = []
        list_cc = []
        for mb in my_blocks:
            if mb.is_true_constraint():
                list_uc = list_uc + mb.contenu()
            else:
                list_cc = list_cc + mb.contenu()

        for uc in list_uc:
            if self.is_relaxed_usual_constraint[uc] and uc not in list_usual_idx_not_to_release:
                self._inv_relax_single_usual_constraint(uc)
                self.is_relaxed_usual_constraint[uc] = None
        for cc in list_cc:
            if self.is_relaxed_constant[cc] and cc not in list_constant_idx_not_to_release:
                self._inv_relax_single_constant_constraint(cc)
                self.is_relaxed_constant[cc] = False
