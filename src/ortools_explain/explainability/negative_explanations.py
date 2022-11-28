from multiprocessing import Process, Queue
from typing import *
import logging

from ortools.sat.python import cp_model

from ortools_explain.model_indexation.constraints import NamedBlock, UsualConstraint
from .constraints_relaxation import ConstraintsRelaxation
from .consistency import is_consistent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


class NegativeExplanation:
    """
    Class for generating the explanation of an unsolvable model.
    """

    def __init__(
            self,
            model,
            split_choice=3,  # Argument explaining how we split the set of constraints in two
            nb_iis_granularite=1
    ):
        self.model = model
        self.dict_indexing = model.dict_indexing()
        self.nb_iis_granularite = nb_iis_granularite

        self.constraints_relaxation = ConstraintsRelaxation(model)
        self.solver = cp_model.CpSolver()

        self.index_var_assumption_to_block: Dict[int, NamedBlock] = dict()
        self.block_to_index_var_assumption: Dict[NamedBlock, int] = dict()
        self.list_idx_usual_constraints_with_assumptions = []
        self.list_idx_constant_constraints_with_assumptions = []
        self.variables_ajoutees = []

        self.list_already_found_iis = []

        self.split_choice = split_choice

    def is_infeasible(self) -> bool:
        """Before searching for IIS, checks once and for all that the model is not feasible"""
        self.model.Proto().ClearField("objective")
        return not is_consistent(self.model, self.solver)

    def find_one_iis_combined(self) -> [NamedBlock]:
        """Launches QuickXplain and Sufficient Assumption on two separate threads
        Returns the first result"""

        def thread_qx():
            logger.debug('Launching QuickXplain')
            res = self.find_one_iis_auto_zoom_quick_xplain()
            logger.debug('QuickXplain is done')
            return res

        def thread_sa():
            logger.debug('Launching Sufficient Assumption')
            res = self.find_one_iis_sufficient_assumption()
            logger.debug('Sufficient Assumption is done')
            return res

        class CustomProcess(Process):
            def __init__(self, tache, queue_de_retour):
                Process.__init__(self)
                self.tache = tache
                self.queue_de_retour = queue_de_retour

            def run(self):
                resultat = self.tache()
                self.queue_de_retour.put(resultat)

        queue_pour_le_resultat = Queue()
        liste_de_process = [CustomProcess(tache, queue_pour_le_resultat) for tache in [thread_qx, thread_sa]]

        for idx, process in enumerate(liste_de_process):
            process.start()

        resultat = queue_pour_le_resultat.get()

        for process in liste_de_process:
            process.terminate()

        return resultat

    def find_one_iis_sufficient_assumption(self) -> [NamedBlock]:
        """
        Finds one IIS with the Sufficient Assumption way
        Also uses a little Quick Xplain because SA does not handle constant constraints
        """
        if not self.is_infeasible():
            logger.info('Problem is solvable')
            return None

        mon_iis_uc = self._find_one_iis_and_zoom('sufficient_assumption')
        if not mon_iis_uc:
            mon_iis_uc = []

        logger.info('IIS found with sufficient assumption :')
        logger.info(mon_iis_uc)
        logger.info('Reducing the IIS with quick xplain... ')

        self.constraints_relaxation.relax_all_real_constraints_but(mon_iis_uc)

        # Finds one IIS with sufficient assumption
        mes_blocks = mon_iis_uc + [b for b in self.dict_indexing.return_all_blocks_as_named() if not b.is_true_constraint()]
        mon_iis_complet = self._find_one_iis_and_zoom('quick_xplain', mes_blocks)

        logger.info('Final IIS:')
        logger.info(mon_iis_complet)

        return mon_iis_complet

    def _find_one_iis_and_zoom(self, method: str, all_blocks=None) -> [NamedBlock]:
        """
        Finds one IIS on self.model (some constraints may be released)
        Zooms automatically to return an IIS with the most appropriate granularity
        At the end of this function, relaxed blocks are exactly the same as in the beginning
        """
        if not all_blocks:
            all_blocks = self.dict_indexing.return_all_blocks_as_named()

        # Granularité initiale
        current_iis = self.find_iis_single_call(method, all_blocks)  # Liste de NamedBlock
        if len(current_iis) == 0:
            logger.info('There is a problem within the background blocks')
            return []

        # On relache une fois pour toutes ce qui n'est pas dans l'IIS
        self.constraints_relaxation.relax_all_but(current_iis, method)

        logger.info('Large IIS: ')
        logger.info(current_iis)

        while True:
            longueur_iis = len(current_iis)
            nouv_iis = None
            new_blocks = None
            for nb in current_iis:
                while nb.can_be_split() and not new_blocks:
                    list_nb = nb.split_on_next_dimension()
                    new_blocks = list_nb + [b for b in current_iis if b != nb]
                    logger.debug('New blocks : {}'.format(new_blocks))
                    nouv_iis = self.find_iis_single_call(method, new_blocks)
                    logger.debug('New-scale IIS : {}'.format(nouv_iis))
                    if len(nouv_iis) > longueur_iis + self.nb_iis_granularite - 1:
                        logger.debug('This dimension is not relevant')
                        nb.remove_dim_to_split()
                        nouv_iis = None
                        new_blocks = None
            if not nouv_iis:
                break
            else:
                current_iis = nouv_iis
                self.constraints_relaxation.relax_all_but(current_iis, method)
                logger.info('Current IIS: ')
                logger.info(current_iis)

        logger.info('Final IIS:')
        logger.info(current_iis)
        self.constraints_relaxation.inv_relax_all()
        return current_iis

    def find_one_iis_auto_zoom_quick_xplain(self) -> [NamedBlock]:
        """
        Finds one IIS by zooming automatically with QX method
        Starts by finding MacroBlocks that are in conflict and then zooms in progressively
        """
        if not self.is_infeasible():
            logger.info('Problem is feasible')
            return None

        return self._find_one_iis_and_zoom('quick_xplain')

    def find_set_of_distinct_iis_combined(self) -> [[NamedBlock]]:
        """Launches QuickXplain and Sufficient Assumption on two separate threads
        Returns the first result"""

        def thread_qx():
            logger.debug('Lauching QuickXplain')
            res = self.find_set_of_distinct_iis_qx()
            logger.debug('QuickXplain is done')
            return res

        def thread_sa():
            logger.debug('Launching Sufficient Assumption')
            res = self.find_set_of_distinct_iis_sa()
            logger.debug('Sufficient Assumption is done')
            return res

        class CustomProcess(Process):
            def __init__(self, tache, queue_de_retour):
                Process.__init__(self)
                self.tache = tache
                self.queue_de_retour = queue_de_retour

            def run(self):
                resultat = self.tache()
                self.queue_de_retour.put(resultat)

        queue_pour_le_resultat = Queue()
        liste_de_process = [CustomProcess(tache, queue_pour_le_resultat) for tache in [thread_qx, thread_sa]]

        for idx, process in enumerate(liste_de_process):
            process.start()

        resultat = queue_pour_le_resultat.get()

        for process in liste_de_process:
            process.terminate()

        return resultat

    def find_set_of_distinct_iis_qx(self) -> [[NamedBlock]]:
        """
        Returns a set of IIS that do not intersect each other, with QX method
        """
        if not self.is_infeasible():
            return None

        mes_iis = []
        self.list_already_found_iis = []
        on_continue = True

        while on_continue:
            logger.info('Searching for a new IIS... ')

            # On relache les contraintes des IIS déjà trouvés
            for iis in mes_iis:
                self.constraints_relaxation.relax_list_blocks(iis)

            # On vérifie qu'on a encore au moins un conflit
            status = is_consistent(self.model, self.solver)

            if status:
                logger.info('No more IIS')
                on_continue = False
            else:
                nouvel_iis = self._find_one_iis_and_zoom('quick_xplain')
                if len(nouvel_iis) == 0:
                    on_continue = False
                else:
                    mes_iis.append(nouvel_iis)
                    self.list_already_found_iis.append(nouvel_iis)

        self.constraints_relaxation.inv_relax_all()
        return mes_iis

    def find_set_of_distinct_iis_sa(self) -> [[NamedBlock]]:
        """
        Returns a set of IIS that do not intersect each other, with SA method
        """
        if not self.is_infeasible():
            return None

        mes_iis = []
        self.list_already_found_iis = []
        on_continue = True

        while on_continue:
            logger.info('Searching for a new IIS... ')

            # On relache les contraintes des IIS déjà trouvés
            for iis in mes_iis:
                self.constraints_relaxation.relax_list_blocks(iis)

            # On vérifie qu'on a encore au moins un conflit
            status = is_consistent(self.model, self.solver)

            if status:
                logger.info('No more IIS')
                on_continue = False
            else:
                mon_iis_uc = self._find_one_iis_and_zoom('sufficient_assumption')

                # On relache les contraintes des IIS déjà trouvés
                for iis in mes_iis:
                    self.constraints_relaxation.relax_list_blocks(iis)

                logger.debug('IIS found with sufficient assumption:')
                logger.debug(mon_iis_uc)

                if not mon_iis_uc:
                    mon_iis_uc = []

                self.constraints_relaxation.relax_all_real_constraints_but(mon_iis_uc)

                # Finds one IIS with sufficient assumption
                mes_blocks = mon_iis_uc + [b for b in self.dict_indexing.return_all_blocks_as_named() if not b.is_true_constraint()]
                mon_iis_complet = self._find_one_iis_and_zoom('quick_xplain', mes_blocks)

                logger.debug('Final IIS:')
                logger.debug(mon_iis_complet)

                if len(mon_iis_complet) == 0:
                    on_continue = False
                else:
                    mes_iis.append(mon_iis_complet)
                    self.list_already_found_iis.append(mon_iis_complet)

        self.constraints_relaxation.inv_relax_all()
        return mes_iis

    def split(self, C: List[NamedBlock], choice: int):
        # Attention à la méthode de découpage
        # S'il n'y a pas au moins un block dans chaque partie on va boucler à l'infini
        """
        Function that split the list C of constraint's blocks into two list of constraint's blocks
        Beware if implementing new splitting methods. If the method occasionally return one empty list, quick_xplain will enter an infinite loop
        :param C: (List[Block]) the list of constraint's blocks
        :param choice: (int) 1 to "isolate" the first block
                              2 to split "by half"
                              3 to split in two and balance the size of both parts
        :return: C_1, C_2: (List[Block]) two lists of constraint's blocks
        """
        if choice == 1:
            return [C[0]], C[1:]
        if choice == 2:
            k = int(len(C) / 2)
            return C[:k], C[k:]
        if choice == 3:
            nb_blocs = len(C)
            if nb_blocs < 2:
                return self.split(C, 2)

            mes_tailles = [b.size() for b in C]
            max_taille = sum(x for x in mes_tailles)

            """On fait un mini problème d'optimisation pour découper notre problème en deux parties égales"""
            model = cp_model.CpModel()
            solver2 = cp_model.CpSolver()
            solver2.parameters.max_time_in_seconds = 60
            affectations = {j: model.NewBoolVar("affect_{}".format(j)) for j in range(nb_blocs)}
            ecart_optimal = model.NewIntVar(0, max_taille, 'bloc_min')
            model.Add((2 * sum([affectations[j] * mes_tailles[j] for j in range(nb_blocs)]) - max_taille) <= ecart_optimal)
            model.Add((max_taille - 2 * sum([affectations[j] * mes_tailles[j] for j in range(nb_blocs)])) <= ecart_optimal)
            model.Minimize(ecart_optimal)
            res = solver2.Solve(model)

            if res == cp_model.OPTIMAL:
                bloc_1 = [C[j] for j in range(nb_blocs) if solver2.Value(affectations[j])]
                bloc_2 = [C[j] for j in range(nb_blocs) if not solver2.Value(affectations[j])]
                if len(bloc_2) == 0 or len(bloc_1) == 0:
                    return self.split(C, 2)
                return bloc_1, bloc_2
            else:
                return self.split(C, 2)

    def add_var_assumptions(self, all_blocks):
        """
        Function that adds assumption variables
        """
        for block in all_blocks:
            if block.is_true_constraint():
                nom_block = block.name()
                block_constraints = block.contenu()

                if block not in self.block_to_index_var_assumption:
                    b_var = self.model.NewBoolVar(nom_block)
                    b_var_ix = b_var.Index()
                    self.variables_ajoutees.append(b_var_ix)
                    self.index_var_assumption_to_block[b_var.Index()] = block
                    self.block_to_index_var_assumption[block] = b_var.Index()
                    self.model.Proto().assumptions.extend([b_var.Index()])

                for single_constraint in block_constraints:
                    if isinstance(single_constraint, UsualConstraint):
                        list_idx = single_constraint.list_idx()
                        for idx in list_idx:
                            if idx in self.list_idx_usual_constraints_with_assumptions:
                                logger.error("Trying to assign an assumption variable to a constraint which already has one")
                                raise IndexError

                            if idx in self.dict_indexing.get_list_usual_inamovible() or self.constraints_relaxation.is_relaxed_usual_constraint[single_constraint]:
                                continue

                            index_b_var = self.block_to_index_var_assumption[block]
                            self.model.Proto().constraints[idx].enforcement_literal.extend([index_b_var])
                            self.list_idx_usual_constraints_with_assumptions.append(idx)

    def inv_add_var_assumptions(self):
        """Function that removes ALL assumption variables"""
        # On enlève les variables assumptions présentes devant les contraintes
        for idx_constraints in self.list_idx_usual_constraints_with_assumptions:
            self.model.Proto().constraints[idx_constraints].enforcement_literal.pop(0)

        self.model.Proto().assumptions[:] = []

        # On supprime les variables d'assumption
        for ind in sorted(self.variables_ajoutees, reverse=True):
            self.model.Proto().variables.pop(ind)

        # On remet à jour notre "mémoire"
        self.index_var_assumption_to_block = dict()
        self.block_to_index_var_assumption = dict()
        self.list_idx_usual_constraints_with_assumptions = []
        self.variables_ajoutees = []

    def sufficient_assumption(self, all_blocks) -> Union[List[NamedBlock], None]:
        """
        Function that returns an iis with the Or-tools function : "sufficient_assumptions_for_infeasibility"
        :return: explanations: (List[Block]) a list of blocks corresponding to an IIS
        """
        logger.info("Launching sufficient_assumption")
        self.add_var_assumptions(all_blocks)
        self.solver.parameters.num_search_workers = 1

        # On vérifie qu'on a encore au moins un conflit
        status = is_consistent(self.model, self.solver)
        if status:
            self.inv_add_var_assumptions()
            return None
        else:
            assumptions = self.solver.ResponseProto().sufficient_assumptions_for_infeasibility
            explanations = set(self.index_var_assumption_to_block[index_b_var] for index_b_var in assumptions)
            self.inv_add_var_assumptions()
            return list(explanations)

    def quick_xplain(self, all_blocks) -> Union[List[NamedBlock], None]:
        """
        Function that implements the QuickXplain's algorithm
        :return: result: (List[Block]) a list of blocks corresponding to an IIS
        """
        logger.info("Launching quick_xplain")

        # list_of_usual_constraints_already_released = copy(self.constraints_relaxation.is_relaxed_usual_constraint)
        # list_of_constant_constraints_already_released = copy(self.constraints_relaxation.is_relaxed_constant)

        list_of_usual_constraints_already_released = {x: True for x in self.constraints_relaxation.is_relaxed_usual_constraint if self.constraints_relaxation.is_relaxed_usual_constraint[x]}
        list_of_constant_constraints_already_released = {x: True for x in self.constraints_relaxation.is_relaxed_constant if self.constraints_relaxation.is_relaxed_constant[x]}

        logger.debug('Relaxed constraints :')
        logger.debug(list(list_of_usual_constraints_already_released.keys()) + ['c_{}'.format(cons) for cons in list_of_constant_constraints_already_released.keys()])

        def QX(delta_bool: bool, remaining_constraints: List[NamedBlock], i_rec=1) -> List[NamedBlock]:
            """
            Core of the recursive function of QuickXplain's algorithm
            :param delta_bool: (bool) True if a constraint has just been activated, or False
            :param remaining_constraints: (List[Block]) remaining constraints for IIS research
            :param i_rec: (int) an iterator
            :return: delta_1 + delta_2: (List[Block]) a list of blocks corresponding to an IIS
            """
            logger.debug("\t" * i_rec + "QX: remaining constraints %s" % remaining_constraints)

            if delta_bool and not is_consistent(self.model, self.solver):
                logger.debug("\t" * i_rec + "QX: infeasible")
                return []
            if len(remaining_constraints) == 1:
                logger.debug("\t" * i_rec + "QX: only one remaining constraint")
                return remaining_constraints

            logger.debug("\t" * i_rec + "QX: feasible")

            C_1, C_2 = self.split(remaining_constraints, choice=self.split_choice)
            logger.debug("\t" * i_rec + "C1: %s, C2: %s" % (C_1, C_2))

            self.constraints_relaxation.inv_relax_list_blocks(C_1, list_usual_idx_not_to_release=list_of_usual_constraints_already_released,
                                                              list_constant_idx_not_to_release=list_of_constant_constraints_already_released)
            logger.debug("\t"*i_rec + "QX: Activate C1 : %s" % C_1)
            logger.debug("\t"*i_rec + "QX: Searches in C_2 after having reactivated C_1 : %s" % C_1)
            delta_2 = QX(delta_bool=True, remaining_constraints=C_2, i_rec=i_rec + 1)
            logger.debug("\t"*i_rec + "QX: Delta_2 : %s" % delta_2)
            self.constraints_relaxation.relax_list_blocks(C_1)
            logger.debug("\t"*i_rec + "QX: Deactivate C1 : %s" % C_1)

            if delta_2 and len(delta_2) > 0:
                logger.debug("\t"*i_rec + "QX: Activate Delta_2 : %s" % delta_2)
                self.constraints_relaxation.inv_relax_list_blocks(delta_2, list_usual_idx_not_to_release=list_of_usual_constraints_already_released,
                                                                  list_constant_idx_not_to_release=list_of_constant_constraints_already_released)
                logger.debug("\t"*i_rec + "QX: Searches in C_1 after having activated D2")
                delta_1 = QX(delta_bool=True, remaining_constraints=C_1, i_rec=i_rec + 1)
                logger.debug("\t" * i_rec + "QX: Delta_1 : %s" % delta_1)

                logger.debug("\t"*i_rec + "QX: Deactivate Delta_2 : %s" % delta_2)
                self.constraints_relaxation.relax_list_blocks(delta_2)
            else:
                logger.debug("\t"*i_rec + "QX: Searches in C_1")
                delta_1 = QX(delta_bool=False, remaining_constraints=C_1, i_rec=i_rec + 1)
                logger.debug("\t" * i_rec + "QX: Delta_1 : %s" % delta_1)

            logger.debug("\t"*i_rec + "QX: Delta_1 + Delta_2 : %s" % (delta_1 + delta_2))

            return delta_1 + delta_2

        if is_consistent(self.model, self.solver):
            logger.error('Quick Xplain was called but model is feasible')
            return None

        # all_blocks = [block for block in self.dict_indexing.splits_on_granularity()]

        logger.debug('List of blocks:')
        logger.debug(all_blocks)

        if len(all_blocks) == 0:
            return []

        logger.debug("QX: Deactivates all constraints: %s" % all_blocks)
        self.constraints_relaxation.relax_list_blocks(all_blocks)
        result = QX(delta_bool=True, remaining_constraints=all_blocks)
        logger.debug("QX: Reactivates all constraints: %s" % all_blocks)
        self.constraints_relaxation.inv_relax_list_blocks(all_blocks,
                                                          list_usual_idx_not_to_release=list_of_usual_constraints_already_released,
                                                          list_constant_idx_not_to_release=list_of_constant_constraints_already_released)
        return result

    def find_iis_single_call(self, method: str, all_blocks) -> Union[List[NamedBlock], None]:
        """
        Function that searches for an IIS according to the chosen method (quick_xplain or sufficient_assumption)
        :return: result: (List[NamedBlock]) a list of blocks corresponding to an IIS
        """
        result = None
        if method == "quick_xplain":
            result = self.quick_xplain(all_blocks)
            logger.info("End of quick_xplain")
        elif method == "sufficient_assumption":
            result = self.sufficient_assumption(all_blocks)
            logger.info("End of sufficient_assumption")

        return sorted(result)
