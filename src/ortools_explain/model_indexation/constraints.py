"""
Module defining different private classes that enable us to track constraints
"""
from abc import ABC, abstractmethod
from functools import total_ordering
from typing import *

import logging

logger = logging.getLogger(__name__)
DEBUG_EXPLAINATIONS = False


class ModelisationError(Exception):
    """Raised when the user makes a mistake while creating a ModelWithIndexing"""
    pass


@total_ordering
class ElementaryConstraint (ABC):
    """
    Class defining the smallest possible constraint for the model.

    An Elementary Constraint can encompass several actual constraints, but these constraints will always be enforced or relaxed as a set
    (at any given time, they are either all relaxed or all enforced)

    It contains :

    **constraint**: the name of the general type of constraint

    **dict_dimensions**: the dimensions of the constraint block (dimension key and value on this dimension)
    """

    def __init__(self, constraint, idx, **dict_dimensions):
        """
        """
        self._constraint = constraint
        self._dict_dimensions = dict_dimensions
        self._dimensions = sorted(list(dict_dimensions.keys()))
        self._first_id = idx

    def dimensions(self):
        return self._dimensions

    def name_macroblock(self):
        return self._constraint

    def dim_value(self, dimension):
        return self._dict_dimensions[dimension]

    def __str__(self):
        if not self._dimensions:
            return self._constraint
        else:
            return (
                    self._constraint + "("
                    + ", ".join(["%s = %s" % (k, str(self._dict_dimensions[k])) for k in self._dimensions]) + ")"
            )

    def __lt__(self, other):
        # First we sort by name
        if self.name_macroblock() != other.name_macroblock():
            return self.name_macroblock() < other.name_macroblock()
        # If they have the same name then they have the same dimensions
        for d in self.dimensions():
            if self.dim_value(d) != other.dim_value(d):
                return self.dim_value(d) < other.dim_value(d)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        return str(self)

    def first_id(self) -> int:
        return self._first_id

    @abstractmethod
    def list_idx(self) -> List[int]:
        pass

    def size(self) -> int:
        return len(self.list_idx())


class UsualConstraint(ElementaryConstraint):
    """
    Class defining a single constraint
    The constraint is of the usual type (eg. model.Add(x + y < 2))
    """

    def __init__(self, constraint, idx, **dict_dimensions):
        super().__init__(constraint, idx, **dict_dimensions)
        self._idx = [idx]

    def add_id(self, idx):
        self._idx.append(idx)

    def list_idx(self) -> List[int]:
        return self._idx


class ConstantConstraint(ElementaryConstraint):
    """
    Class defining a single "constant" constraint
    The constraint is in fact a constant (eg. x = model.NewConstant(1))
    """

    def __init__(self, constraint, idx, default_value, default_range, **dict_dimensions):
        super().__init__(constraint, idx, **dict_dimensions)
        self._idx = {idx: {"value": default_value, "range": default_range}}

    def def_value(self, idx):
        return self._idx[idx]["value"]

    def def_range(self, idx):
        return self._idx[idx]["range"]

    def add_id(self, idx, default_value, default_range):
        self._idx[idx] = {"value": default_value, "range": default_range}

    def print_specifique(self):
        return str(self)

    def list_idx(self) -> List[int]:
        return list(self._idx.keys())


class MacroBlock:
    """
    Class defining a set of elementary constraints (eg. pos)
    It contains a set of BlocElementaryConstraint
    A bloc can be projected on various dimensions
    """

    def __init__(self, name, firstConstraint: ElementaryConstraint, is_constraint):
        self._name = name
        self._dimensions = firstConstraint.dimensions()
        if not self._dimensions:
            self._dimensions = []
        if len(self._dimensions) == 0:
            clef = ""
        else:
            clef = tuple([firstConstraint.dim_value(d) for d in self.dimensions()])
        self.list_ec = {clef: firstConstraint}
        self._is_true_constraint = is_constraint  # True if true constraint, False if constant
        self._explanations = dict()
        self._dim_a_ignorer = []

    def contenu(self) -> [ElementaryConstraint]:
        return list(self.list_ec.values())

    def to_named_block(self):
        dim_peut_split = [d for d in self.dimensions() if d not in self._dim_a_ignorer]
        return NamedBlock(self, [], dim_peut_split, list(self.list_ec.values()), self._is_true_constraint)

    def name(self) -> str:
        return self._name

    def dimensions(self) -> [str]:
        return self._dimensions

    def do_not_split_on(self, *dim):
        """Indicates that we must not split on this dimension because it does not mean anything for the user"""
        for d in dim:
            if d in self.dimensions() and d not in self._dim_a_ignorer:
                self._dim_a_ignorer.append(d)

    def size(self) -> int:
        return sum(ec.size() for ec in self.list_ec.values())

    def is_true_constraint(self):
        return self._is_true_constraint

    def __repr__(self):
        if self.is_true_constraint():
            return "{} -- dim: {} -- size: {}".format(self.name(), self.dimensions(), self.size())
        else:
            return "{} (CONSTANTES) -- dim: {} -- size: {}".format(self.name(), self.dimensions(), self.size())

    def ajoute_ec(self, nouvelleContrainte: ElementaryConstraint) -> ElementaryConstraint:
        """
        Adds a new constraint to the block after checking that the model is consistent
        """
        # Raises an error if is_true_constraint is not consistent
        if self.is_true_constraint() and isinstance(nouvelleContrainte, ConstantConstraint):
            raise ModelisationError("Le nom {} est utilisé à la fois pour des contraintes et des constantes".format(self.name()))
        if not self.is_true_constraint() and isinstance(nouvelleContrainte, UsualConstraint):
            raise ModelisationError("Le nom {} est utilisé à la fois pour des contraintes et des constantes".format(self.name()))

        # Raises an error if dimensions are not consistent throughout the model for the same block name
        if nouvelleContrainte.dimensions() != self.dimensions():
            raise ModelisationError("Erreur de cohérence sur les dimensions de la contrainte {}. Le modèle a rencontré {} puis {}".format(self.name(),
                                                                                                                                          self.dimensions(), nouvelleContrainte.dimensions()))

        # Overloads the existing constraint if it exists with the same dimensions
        clef = tuple([nouvelleContrainte.dim_value(d) for d in self.dimensions()])
        if clef in self.list_ec:
            vieilleContrainte = self.list_ec[clef]
            if isinstance(vieilleContrainte, UsualConstraint):
                vieilleContrainte.add_id(nouvelleContrainte.first_id())
                return vieilleContrainte
            elif isinstance(vieilleContrainte, ConstantConstraint):
                assert isinstance(nouvelleContrainte, ConstantConstraint)
                new_id = nouvelleContrainte.first_id()
                new_val = nouvelleContrainte.def_value(new_id)
                new_range = nouvelleContrainte.def_range(new_id)
                vieilleContrainte.add_id(new_id, new_val, new_range)
                return vieilleContrainte
            # raise ModelisationError("La contrainte {} pour les dimensions {} est définie deux fois.".format(self.name(), clef))
        else:
            self.list_ec[clef] = nouvelleContrainte
            return nouvelleContrainte

    def project_on_dimensions(self, **dimensions) -> [ElementaryConstraint]:
        """Returns the list of BlocKElementaryConstraints that match the given dimensions"""
        if len(dimensions.keys()) == 0:
            return [ec for ec in self.list_ec.values()]

        for k in dimensions.keys():
            if k not in self._dimensions:
                logger.info("La dimension {} n'existe pas. Les dimensions existantes sont {}.".format(k, self._dimensions))
                return []

        ma_liste = []
        for ec in self.list_ec.values():
            on_garde = True
            for (k, v) in dimensions.items():
                if ec.dim_value(k) != v:
                    on_garde = False
                    pass
                if on_garde:
                    ma_liste.append(ec)
        return ma_liste

    def add_explanations(self, *sentences):
        """
        Adds explanations to the macroblock in natural language
        """
        for s in sentences:
            dimensions_utilisees = []
            for d in self.dimensions():
                clef = '{' + d + '}'
                if clef in s:
                    dimensions_utilisees.append(d)
            mes_dim = '-'
            for d in sorted(dimensions_utilisees):
                mes_dim = mes_dim + d + '-'
            self._explanations[mes_dim] = s

    def get_explanation_for_key(self, key):
        if key in self._explanations:
            return self._explanations[key]
        else:
            return None


@total_ordering
class NamedBlock:
    """
    Class which contains a list of elementary constraints
    The main interest of this class is to be able to name easily this list
    NamedBlock can have different granularities (from entire macroblock to single elementary constraint)
    """

    def __init__(self, mon_macro_block, dim_already_split: [str], dim_not_yet_split: [str], contenu: [ElementaryConstraint], is_true_constraint):
        self._macro_bloc = mon_macro_block
        self._type_bloc = mon_macro_block.name()
        self._dim_already_split = dim_already_split
        self._dim_not_yet_split = dim_not_yet_split
        self._contenu = contenu
        self._is_true_constraint = is_true_constraint  # True if true constraint, False if constant

    def __lt__(self, other):
        # First we sort by name
        if self.type_bloc() != other.type_bloc():
            return self.type_bloc() < other.type_bloc()
        # Then we sort on how precise they have
        if len(self.dimensions()) != len(other.dimensions()):
            return self.dimensions() < other.dimensions()
        # Then we split on the sentence
        return self.elegant_name() < other.elegant_name()

    def __hash__(self):
        return hash(self.name())

    def name(self):
        if len(self.contenu()) == 0:
            return 'Bloc vide'
        if len(self._dim_already_split) == 0:
            return self.type_bloc()
        ec = self.contenu()[0]
        key = None
        for d in self._dim_already_split:
            if key:
                key = "{}, {} = {}".format(key, d, ec.dim_value(d))
            else:
                key = "{} = {}".format(d, ec.dim_value(d))
        return "{} ({})".format(self.type_bloc(), key)

    def contenu(self) -> [ElementaryConstraint]:
        return self._contenu

    def type_bloc(self):
        return self._type_bloc

    def dimensions(self):
        return self._dim_already_split

    def size(self):
        return sum(ec.size() for ec in self._contenu)

    def is_true_constraint(self):
        return self._is_true_constraint

    def can_be_split(self) -> bool:
        return len(self._dim_not_yet_split) > 0

    def to_dict(self) -> Dict:
        """Used for returning a dictionary instead of a NamedBlock object"""
        my_dimensions = {}
        example = self.contenu()[0]
        for d in self._dim_already_split:
            my_dimensions[d] = example.dim_value(d)
        return {"sentence": self.elegant_name(),
                "type": self._type_bloc,
                "dimensions": my_dimensions}

    def elegant_name(self):
        if DEBUG_EXPLAINATIONS:
            return '{} ({})'.format(self.name(), self.size())

        if len(self.contenu()) == 0:
            return 'Bloc vide'
        example = self.contenu()[0]
        mes_dim_actuelles = {}
        for d in self._dim_already_split:
            mes_dim_actuelles[d] = example.dim_value(d)
        ma_clef = '-'
        for d in sorted(list(mes_dim_actuelles.keys())):
            ma_clef = ma_clef + d + '-'

        my_sentence = self._macro_bloc.get_explanation_for_key(ma_clef)
        if not my_sentence:
            return self.name()

        for d in mes_dim_actuelles:
            dim_key = '{' + d + '}'
            my_sentence = my_sentence.replace(dim_key, str(mes_dim_actuelles[d]))

        return my_sentence

    def __repr__(self):
        # return self.name()
        # return '{} ({})'.format(self._name, self.size())
        return self.elegant_name()

    def remove_dim_to_split(self):
        self._dim_not_yet_split = [c for c in self._dim_not_yet_split[1:]]

    def split_on_next_dimension(self):
        """
        Returns this block split in several blocks with smaller granularity
        """
        if len(self._dim_not_yet_split) == 0:
            return None

        nouv_dim = self._dim_not_yet_split[0]
        logger.info('On scinde {} sur la dimension {}'.format(self.name(), nouv_dim))
        dimensions = [d for d in self._dim_already_split] + [nouv_dim]
        remaining_dim = [d for d in self._dim_not_yet_split if d != nouv_dim]

        dico_val = {}
        for ec in self.contenu():
            assert isinstance(ec, ElementaryConstraint)
            key = None
            for d in dimensions:
                if key:
                    key = "{}, {} = {}".format(key, d, ec.dim_value(d))
                else:
                    key = "{} = {}".format(d, ec.dim_value(d))

            if key in dico_val:
                dico_val[key].append(ec)
            else:
                dico_val[key] = [ec]

        liste_blocks_res = []
        for key in dico_val:
            b = NamedBlock(self._macro_bloc, dimensions, remaining_dim, dico_val[key], self._is_true_constraint)
            liste_blocks_res.append(b)

        return liste_blocks_res


class DictIndexConstraints:
    """
    Class for managing the granularity of constraint blocks
    It contains dictionaries that enable us to convert idx into elementary constraint and vice versa
    It keeps in memory the granularity of the model
    """

    def __init__(self):
        self._dict_usual_constraints: Dict[int, UsualConstraint] = dict()
        self._dict_constant_constraints: Dict[int, ConstantConstraint] = dict()
        self._dict_variables = dict()

        self._list_blocks = dict()
        self._unmovable_usual_constraints = []
        self._unmovable_constant_constraints = []

    def print_status(self):
        print('{} contraintes constantes'.format(len(self._dict_constant_constraints.keys())))
        for cc in self._dict_constant_constraints.values():
            print(cc.print_specifique())
        print('{} contraintes usuelles'.format(len(self._dict_usual_constraints.keys())))
        for uc in self._dict_usual_constraints.values():
            print(uc)

    def add_variable(self, variable, variable_idx: int):
        """Remembers a given variable, so that if later we define it as a constant, we know where it is in our Proto
        """
        self._dict_variables[variable] = variable_idx

    def get_variable_idx(self, variable):
        """Returns the idx of a variable in the Proto. Returns -1 if it does not exist."""
        if variable in self._dict_variables:
            return self._dict_variables[variable]
        return -1

    def get_list_variables(self):
        return self._dict_variables.keys()

    def add_usual_constraint(self, block: UsualConstraint, constraint_idx: int):
        """
        Function that is called when the model is being created
        Associates a block to an OR-TOOLS id
        Updates the list of blocks within the model
        :param block: constraint that must be added
        :param constraint_idx: id of the constraints
        """
        # Looks whether this constraint has already been defined
        name_set = block.name_macroblock()
        if name_set not in self._list_blocks:
            self._list_blocks[name_set] = MacroBlock(name_set, block, True)
        else:
            block = self._list_blocks[name_set].ajoute_ec(block)

        # Keeps in memory the link between the elementary constraint and the idx
        self._dict_usual_constraints[constraint_idx] = block

    def add_constant_constraint(self, block: ConstantConstraint, constraint_idx: int):
        """
        Function that is called when the model is being created
        Associates a block to an OR-TOOLS id
        Updates the list of blocks within the model
        Function that associates to a block the OR-TOOLS constraint "constraint_idx"
        :param block: a single constraint
        :param constraint_idx: index of the constraint in the OR-Tools proto
        """
        # Updates the list of blocs
        name_set = block.name_macroblock()
        if name_set not in self._list_blocks:
            self._list_blocks[name_set] = MacroBlock(name_set, block, False)
        else:
            block = self._list_blocks[name_set].ajoute_ec(block)

        self._dict_constant_constraints[constraint_idx] = block

    def list_usual_constraints(self) -> [UsualConstraint]:
        return list(self._dict_usual_constraints.values())

    def list_constant_constraints(self) -> [ConstantConstraint]:
        res = []
        for mb in self.list_all_macro_blocks():
            if not mb.is_true_constraint():
                res = res + mb.contenu()
        return res

    def list_all_macro_blocks(self) -> [MacroBlock]:
        """Returns the list of macro blocks (eg: "pos", "line", "col", "square" for sudoku)"""
        return list(set(self._list_blocks.values()))

    def list_macro_block_names(self) -> [str]:
        """Returns the list of block names"""
        # return list(set(b.name_associated_constraint() for b in self.list_elementary_blocks()))
        return list(self._list_blocks.keys())

    def get_mb(self, name) -> MacroBlock:
        return self._list_blocks[name]

    def has_mb(self, name) -> bool:
        return name in self._list_blocks

    def has_constant_block(self, idx: int):
        return idx in self._dict_constant_constraints

    def get_constant_block(self, idx: int) -> ConstantConstraint:
        """Converts from constraint to id"""
        return self._dict_constant_constraints[idx]

    def get_usual_block(self, idx: int) -> UsualConstraint:
        """Converts from constraint to id"""
        return self._dict_usual_constraints[idx]

    def define_as_unamovible(self, name: str, **dict_dimensions):
        """Remembers that some constraints cannot be touched"""
        if name not in self.list_macro_block_names():
            logger.info("Le bloc de contraintes {} n'existe pas dans notre modèle".format(name))
            return
        else:
            ma_liste = self._list_blocks[name].project_on_dimensions(dict_dimensions)
            is_true_constraint = self._list_blocks[name].is_true_constraint()

            if is_true_constraint:
                for elt in ma_liste:
                    idx = elt.list_idx()
                    if idx not in self._unmovable_usual_constraints:
                        self._unmovable_usual_constraints.append(idx)
            else:
                for elt in ma_liste:
                    idx = elt.list_idx()
                    if idx not in self._unmovable_constant_constraints:
                        self._unmovable_constant_constraints.append(idx)

    def reset_unamovible(self):
        """Resets the entire list of unamovible constraints"""
        self._unmovable_usual_constraints.clear()
        self._unmovable_constant_constraints.clear()

    def define_as_amovible(self, name: str, **dict_dimensions):
        """Removes the indication that one constraint cannot be touched"""
        if name not in self.list_macro_block_names():
            logger.info("Le bloc de contraintes {} n'existe pas dans notre modèle".format(name))
            return
        else:
            ma_liste = self._list_blocks[name].project_on_dimensions(dict_dimensions)
            is_true_constraint = self._list_blocks[name].is_true_constraint()

            if is_true_constraint:
                for elt in ma_liste:
                    idx = elt.list_idx()
                    if idx in self._unmovable_usual_constraints:
                        self._unmovable_usual_constraints.remove(idx)
            else:
                for elt in ma_liste:
                    idx = elt.list_idx()
                    if idx in self._unmovable_constant_constraints:
                        self._unmovable_constant_constraints.remove(idx)

    def get_list_usual_inamovible(self):
        return self._unmovable_usual_constraints

    def get_list_constant_inamovible(self):
        return self._unmovable_constant_constraints

    def return_all_blocks_as_named(self) -> [NamedBlock]:
        return [b.to_named_block() for b in self.list_all_macro_blocks()]

    def print(self):
        print("Mes blocs de contraintes : ")
        for b in self._list_blocks.values():
            print('-- ' + str(b))
