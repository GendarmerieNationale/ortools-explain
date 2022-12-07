import logging
from abc import ABC, abstractmethod
from typing import List

logger = logging.getLogger(__name__)


class LNS_Variables_Choice(ABC):
    @abstractmethod
    def variables_to_block(self) -> List:
        """
        At each call, must return the next variables to *freeze*
        """
        pass

    @abstractmethod
    def nb_remaining_iterations(self) -> int:
        """Returns the remaining number of loops to go"""
        pass


class LNS_Variables_Choice_Across_One_Dim(LNS_Variables_Choice):
    def __init__(self, variables_to_dim, window_size: int, step: int, nb_iterations: int = 1):
        self.variables_to_dim = variables_to_dim
        self.taille_fenetre = window_size
        self.pas_decalage = step
        self.nb_iterations = nb_iterations

        liste_val_ma_dim = sorted(list(set(self.variables_to_dim.values())))
        nb_val = len(liste_val_ma_dim)
        self.mes_intervalles = []
        curs = 0
        while curs < nb_val:
            val_init = liste_val_ma_dim[curs]
            proch_valeur = val_init + step
            proch_curs = None
            for j in range(curs + 1, nb_val):
                if liste_val_ma_dim[j] >= proch_valeur:
                    proch_curs = j
                    break

            nouv_interv = []
            borne_sup_stricte = val_init + window_size
            fin_liste_atteinte = False
            while True:
                if curs == nb_val:
                    fin_liste_atteinte = True
                    break
                elif liste_val_ma_dim[curs] < borne_sup_stricte:
                    nouv_interv.append(liste_val_ma_dim[curs])
                    curs += 1
                else:
                    break
            self.mes_intervalles.append(nouv_interv)
            logger.info("New interval found: {}".format(nouv_interv))
            if fin_liste_atteinte:
                break
            else:
                curs = proch_curs

        logger.info("Final interval list: {}".format(self.mes_intervalles))
        nb_fenetres = len(self.mes_intervalles)

        if nb_fenetres < 2:
            logger.info("LNS is useless with these parameters")

        self.iteration_courante = 0
        self.idx_fenetre_courante = 0
        self.idx_problem = 0

        self.nb_tot_pbs = nb_iterations * nb_fenetres

    def variables_to_block(self):
        fen = self.mes_intervalles[self.idx_fenetre_courante]
        list_var_a_bloquer = [v for v in self.variables_to_dim if self.variables_to_dim[v] not in fen]
        self.idx_fenetre_courante += 1
        self.idx_problem += 1

        if self.idx_fenetre_courante == len(self.mes_intervalles):
            self.iteration_courante += 1
            self.idx_fenetre_courante = 0

        return list_var_a_bloquer

    def nb_remaining_iterations(self):
        return self.nb_tot_pbs - self.idx_problem
