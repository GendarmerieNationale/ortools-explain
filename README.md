# Installation

Le module ortools-explain est un module visant à compléter le cpsolver du module OR Tools de Google.
Aucune librairie python autre qu'**ortools** n'est nécessaire pour faire fonctionner le module.

La documentation complète du module (en anglais) est disponible à : https://datalab-stsisi.github.io/

# Présentation

Le module ortools-explain vous permet de réaliser plus facilement certaines opérations sur les modèles CP :

* Implémentation sur un même modèle de plusieurs objectifs (optimisation séquentielle ou combinée)
* Implémentation de contraintes relaxables (contraintes facultatives qui accordent des bonus si elles sont respectées)
* Explication de l'infaisabilité sur les problèmes infaisables
* Explication locale de la solution sur les problèmes faisables
* Restitution des explications en langage naturel
* Optimisation locale simplifiée (algorithme LNS)

Le package **examples** contient deux fichiers avec des exemples utilisant les principales fonctions du module.

## SuperModel

La classe **SuperModel** est une surcouche de la classe cp_model de ortools qui permet de nommer des contraintes (pour l'explicabilité négative) et
de définir des objectifs multiples, y compris des objectifs qui consistent à activer ou non une contrainte.

### Explicabilité négative

#### *Add* et *AddConstant*

Les fonctions *Add* et *AddConstant* sont appelées lors de la construction du modèle.
Elles servent à indiquer au module la structure du problème afin qu'il puisse renvoyer les parties du problèmes
qui sont en conflit si le problème est infaisable.

Les contraintes sont définies de la façon suivante : 

```
model.Add(X[i, j1] + X[i, j2] == 1, "unicité", personne=i)
```

où "unicité" est le **type** de la contrainte, et "personne=i" est la seule **dimension**, dont la clef est "personne" et la valeur est i.
Les contraintes peuvent avoir 0, 1 ou plusieurs dimensions.

Les lignes ci-dessous illustrent des appels corrects à *Add* :

```
# Le type et les dimensions ne sont pas obligatoires
# Les contraintes sans type sont appelées "background blocks" et ne peuvent jamais être désactivées
model.Add(X[i, j1] + X[i, j2] == 1)

# Les dimensions ne sont pas obligatoires
model.Add(X[i, j1] + X[i, j2] == 1, "unicité")

model.Add(X[i, j1] + X[i, j2] == 1, "unicité", personne=i)
model.Add(X[i, j1] + X[i, j2] == 1, "unicité", personne=i, premier_jour=j1)
model.Add(X[i, j1] + X[i, j2] == 1, "unicité", personne=i, premier_jour=j1, deuxieme_jour=j2)
model.Add(X[i, j1] + X[i, j2] == 1, "unicité", personne=i, jours=('{}-{}'.format(j1, j2)))
model.Add(X[i, j1] + X[i, j2] == 1, "unicité", personne=i+1)
model.Add(X[i, j1] + X[i, j2] == 1, "unicité", personne='personne n°%d'%i)
```

Les constantes sont des contraintes spécifiques. Dans le module OR Tools, les constantes sont définies via un appel
à *model.NewConstant(valeur)*.

Dans notre module, les constantes sont définies de façon très similaires aux contraintes, avec la fonction *AddConstant*.
Les constantes définies avec *AddConstant* doivent toujours être d'abord définies en tant que variables.

```
X[i, j, k] = model.NewBoolVar('position_{}_{}_{}'.format(i, j, k))
model.AddConstant(X[i, j, k], 0, "position_initiale", personne=i, mission=j)
```

#### Utilisations correctes et incorrectes

```
# --- CONSTANTES ---

# CORRECT - les constantes peuvent aussi être définies en tant que background blocks (sans type et sans dimensions)
model.AddConstant(x[i, j], 0)

# MAUVAISE UTILISATION - l'utilisation de NewConstant comme dans le module OR Tools initial fonctionne
# mais risque de produire des explications peu précises pour les problèmes infaisables. 
x[i, j] = model.NewConstant(0)
# Utiliser à la place l'implémentation précédente

# --- DIMENSIONS ---

# INCORRECT - on ne peut pas donner de dimensions sans donner de type
model.Add(X[i, j1] + X[i, j2] == 1, personne=i)

# INCORRECT - les valeurs des dimensions ne peuvent pas contenir '{' ou '}'
model.Add(X[i, j1] + X[i, j2] == 1, "mon_type", ma_dimension = "}valeur_stupide{")

# INCORRECT - les valeurs des dimensions doivent être "hashables" (par exemple str, int)
# et en particulier ne peuvent pas être des objets modifiables comme une liste ou un dictionnaire
model.Add(X[i, j1] + X[i, j2] == 1, "mon_type", ma_dimension = "[j1, j2]")

# CORRECT - les valeurs des dimensions peuvent être de n'importe quel type tant que celui-ci est "hashable"
model.Add(X[i, j1] + X[i, j2] == 1, "mon_type", ma_dimension = "(j1, j2)")

# --- COMBINAISON DE CONTRAINTES ---

# CORRECT - plusieurs contraintes peuvent avoir le même type et les mêmes dimensions
model.Add(a + b > 0, "somme", id=a)
model.Add(a + c > 0, "somme", id=a)

# CORRECT - une variable peut être définie comme constante avec deux valeurs différentes.
# (Ce type de problème est infaisable de manière évidente mais la modélisation en elle-même est correcte)
model.AddConstant(pos[0, 0, 1], 1, "position_à_1")
model.AddConstant(pos[0, 0, 1], 0, "position_à_0")

# CORRECT - plusieurs constantes peuvent être déclarées avec le même type et les mêmes dimensions même si la valeur n'est pas la même
model.AddConstant(pos[0, 0, 1], 1, "position", x=0, y=0)
model.AddConstant(pos[0, 0, 2], 0, "position", x=0, y=0)

# INCORRECT - des contraintes avec le même type doivent avoir les mêmes dimensions (mêmes clefs)
model.Add(a == b, "mon_type")
model.Add(c == d, "mon_type", dim1=c)

# INCORRECT - le même type ne peut pas être donné à une contrainte normale (déclarée avec Add)
# et une contrainte de constante (déclarée avec AddConstant)
model.Add(a == b, "mon_type")
model.AddConstant(var, value, "mon_type")
```

#### *AddExplanation*

*AddExplanation* permet de lier une contrainte à une explication en langage naturel qu'un utilisateur pourra comprendre.

Si on considère les contraintes suivantes :
```
for i in liste_personnes:
    for j in liste_jours:
        model.Add(X[i, j, k1] + X[i, j, k2] <= 1, "une_seule_mission_par_personne", personne=i, jour=j)
```

Selon les cas, le solveur pour l'infaisabilité pourra renvoyer l'une des explications suivantes : 
```
>> "une_seule_mission_par_personne"
>> "une_seule_mission_par_personne" (personne= p1)
>> "une_seule_mission_par_personne" (jour= j2)
>> "une_seule_mission_par_personne" (personne= p1, jour= j2)
```

En ajoutant les lignes suivantes :
```
model.AddExplanation("une_seule_mission_par_personne",
"Une personne ne peut pas avoir plus d'une mission par jour",
"{personne} ne peut pas avoir plus d'une mission par jour",
"Une personne ne peut pas avoir plus d'une mission le jour {jour}",
"{personne} ne peut pas avoir plus d'une mission le jour {jour}")
```

Les explications renvoyées deviendront :
```
>> "Une personne ne peut pas avoir plus d'une mission par jour",
>> "p1 ne peut pas avoir plus d'une mission par jour",
>> "Une personne ne peut pas avoir plus d'une mission le jour j2",
>> "p1 ne peut pas avoir plus d'une mission le jour j2"
```

## Modélisation multi-objectifs

Ce module vous permet d'implémenter plusieurs objectifs dans un problème d'optimisation, là où OR Tools n'en autorise qu'un.

Les objectifs partiels sont définis via les fonctions suivantes : 

```
# Ajoute une contrainte facultative, qui augmentera le score d'optimisation (ici de 50 points) si elle est respectée
model.AddRelaxableConstraint(assignment[i_1, j] + assignment[i_2, j] == 1, idx= "ops_need", coef= 50, priority= 1)

# Ajoute un objectif à maximiser
model.AddMaximumObjective(goal_1, idx= "premier_objectif", priority= 2)

# Ajoute un objectif à minimiser
model.AddMinimumObjective(2 * goal_2, idx= "deuxieme_objectif", priority= 2)
```

Le module traitera l'optimisation par ordre croissant de priorité. Plusieurs objectifs peuvent être définis avec la même priorité,
auquel cas ils seront combinés.

### Utilisations correctes et incorrectes

```
# INCORRECT - coef doit être un entier strictement positif
model.AddRelaxableConstraint(x[i_1, j] + x[i_2, j] == 2, idx= "punir le binomage", coef= -50, priority= 1)
# CORRECTION POSSIBLE :
model.AddRelaxableConstraint(x[i_1, j] + x[i_2, j] != 2, idx= "récompenser le non binomage", coef= 50, priority= 1)

# TOUS INCORRECTS - priority doit être un entier strictement positif
model.AddMaximumObjective(my_sum, idx= "combinaison", priority= 0)
model.AddMaximumObjective(my_sum, idx= "combinaison", priority= 0.5)
model.AddMaximumObjective(my_sum, idx= "combinaison", priority= -1)

# INCORRECT - Les idx doivent être différents d'un objectif un autre, avec une exception (voir exemple suivant)
model.AddRelaxableConstraint(x[0, 1] + x[0, 2] == 2, idx= "combinaison", coef= 50, priority= 1)
model.AddMaximumObjective(my_sum, idx= "combinaison", priority= 1)

# CORRECT - 2 contraintes relachables peuvent avoir le même idx, à condition qu'elles aient le même coef et la même priorité
model.AddRelaxableConstraint(x[0, 1] + x[0, 2] == 2, idx= "combinaison", coef= 50, priority= 1)
model.AddRelaxableConstraint(x[1, 1] + x[1, 2] == 2, idx= "combinaison", coef= 50, priority= 1)

# INCORRECT - Même idx et priorités différentes
model.AddRelaxableConstraint(x[0, 1] + x[0, 2] == 2, idx= "combinaison", coef= 50, priority= 1)
model.AddRelaxableConstraint(x[1, 1] + x[1, 2] == 2, idx= "combinaison", coef= 50, priority= 2)

# INCORRECT - Même idx et coef différents
model.AddRelaxableConstraint(x[0, 1] + x[0, 2] == 2, idx= "combinaison", coef= 50, priority= 1)
model.AddRelaxableConstraint(x[1, 1] + x[1, 2] == 2, idx= "combinaison", coef= 80, priority= 1)
```

## SuperSolver

Cette classe est une surcouche de cp_model.CpSolver d'OR Tools. Elle permet l'optimisation multi-objectifs, l'optimisation locale et l'explicabilité positive locale.

### Résolution avec SuperSolver

SuperSolver.Solve() renvoie un statut qui est l'un des suivants : 

* UNKNOWN -- similaire à cp_model.UNKNOWN
* MODEL_INVALID -- similaire à cp_model.MODEL_INVALID
* FEASIBLE -- similaire à cp_model.FEASIBLE
* INFEASIBLE -- similaire à cp_model.INFEASIBLE
* OPTIMAL -- similaire à cp_model.OPTIMAL
* OBVIOUS_CONFLICT -- le solveur n'a pas été lancé car un conflit évident a été détecté dès la création du modèle
  (par exemple une variable a été définie deux fois en tant que constante avec deux valeurs différentes)
* NEVER_LAUNCHED -- le solveur n'a encore jamais été lancé (appeler *Solve()*)

#### Exemple

```
my_model = SuperModel()

# Le modèle est créé ici en définissant les variables, en ajoutant les contraintes et les objectifs

# ATTENTION - Contrairement à cp_model.CpSolver, SuperSolver prend le modèle en argument lors de son initialisation
my_solver = SuperSolver(my_model)
status = my_solver.Solve()  # Solve(model) fonctionne aussi, en mimétisme du module OR Tools

if status == Status.OPTIMAL:
    print("L'optimisation a abouti")
    print(my_solver.GetObjectiveValues())

    # Vous pouvez faire des explications locales ici

elif status == Status.FEASIBLE:
    print("Une solution a été trouvée mais n'est pas optimale")

    # Vous pouvez faire des explications locales ici
    # Vous pouvez faire une optimisation locale ici

elif status == Status.OBVIOUS_CONFLICT:
    print("Des conflits évidents ont été détectés dans le modèle")
    my_conflicts = my_model.list_obvious_conflicts()
    for conflict in my_conflicts:
        print(conflict.write_conflict(my_model))

elif status == Status.INFEASIBLE:
    print("Il y a des conflits dans le modèle")
    conflicts = my_solver.ExplainWhyNoSolution()
    print(conflicts)
```

  ### Optimisation locale

SuperSolver contient une fonction LNS qui permet d'améliorer la solution trouvée par le solveur en cherchant des améliorations locales.

Pour cela, une partie du problème est fixée à sa valeur actuelle et l'optimisation n'est autorisée que dans le reste du problème.

Pour utiliser LNS, vous devez créer une classe qui hérite de la classe abstraite *LNS_Variables_Choice*.
Les classes qui héritent de *LNS_Variables_Choice* doivent définir deux méthodes :

* **variables_to_block()** renvoie à chaque appel les variables qui seront fixées à leur valeur actuelle 
* **nb_remaining_iterations()** renvoie à chaque appel le nombre restant d'optimisations locales

#### Exemple - Planification

Nous prenons comme exemple un problème de planification où *X[i, j, k]* sont des variables booléennes
telles que *X[i, j, k] == 1* signifie que la personne i, le jour j, est affectée à la mission k.

```
my_model = SuperModel()

for i in liste_personnes:
    for j in liste_jours:
        for k in liste_missions:
            X[i, j, k] = my_model.NewBoolVar("prend_la_mission_{}_{}_{}".format(i, j, k)")

# Les contraintes sont ajoutées ici
# Les objectifs sont ajoutés ici

my_solver = SuperSolver(my_model)
```

Ici le solveur aura atteint une solution qui ne sera pas l'optimum absolu.

Une façon d'optimiser est de décider de n'optimiser qu'entre les 3 personnes avec le pire planning
et les 3 personnes avec le meilleur planning, et de faire cela 10 fois de suite :

```
# On définit ici la stratégie LNS
class LNS_Equite(LNS_Variables_Choice):
    def __init__(self, X, nb_iterations):
        self.my_variables = X
        self.nb_iterations = nb_iterations
        self.nb_done_optim = 0

    def trie_par_qualite_du_planning(self) -> [int]:
        # Renvoie la liste des personnes classée par ordre croissant de qualité du planning dans la solution actuelle
        pass

    def variables_to_block(self):
        self.nb_done_optim += 1
        people_sorted = self.trie_par_qualite_du_planning()
        people_not_to_change = people_sorted[3:-3]
        variables_to_block = [X[i, j, k] for (i, j, k) in product(people_not_to_change, liste_jours, liste_missions)

    def nb_remaining_iterations(self):
        return self.nb_iterations - self.nb_done_optim

# Ici on l'utilise pour l'optimisation
if my_solver.status() == Status.FEASIBLE:
    my_variables = [X[i, j, k] for (i, j, k) in product(liste_personnes, liste_jours, liste_missions)]
    lns_strategy = LNS_Equite(my_variables, nb_iterations= 10)
    my_solver.LNS(lns_strategy, max_time_lns= 300)

```

Une autre façon d'optimiser et d'optimiser localement sur des journées consécutives. On choisit de travailler sur des fenêtres de 5 jours et d'optimiser localement.
On choisit de déplacer cette fenêtre de 5 jours - ici par des décalages de 2 jours - sur toute l'étendue du problème pour optimiser partout.

Ainsi si liste_jours est [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], l'optimisation se fera sur les fenêtres suivantes :
[0, 1, 2, 3, 4], [2, 3, 4, 5, 6], [4, 5, 6, 7, 8], [6, 7, 8, 9]

Le module contient une fonction intégrée qui permet d'implémenter directement ce type d'optimisation (fenêtre mobile sur une seule dimension) :

  ```
  if my_solver.status() == Status.FEASIBLE:
      # On crée un dictionnaire qui relie chaque variable à sa valeur sur la dimension qui nous intéresse (ici le jour)
      dict_for_lns = {X[i, j, k]: j for (i, j, k) in product(liste_personnes, liste_jours, liste_missions)}
      # On définit simplement la stratégie LNS avec la fonction intégrée au module
      lns_strategy = LNS_Variables_Choice_Across_One_Dim(dict_for_lns, window_size= 5, step= 2, nb_iterations= 1)

      my_solver.LNS(lns_strategy, max_time_lns= 300)
  ```

  ### Explicabilité locale

La classe SuperSolver permet de demander pourquoi une variable a été assignée à telle ou telle valeur, avec *ExplainValueOfVar*.
Par exemple si le solveur renvoie une solution où la variable booléenne *X[0, 0]* a été assignée à 1, vous pouvez demander pourquoi avec :

  ```
  print(solver.ExplainValueOfVar(X[0, 0]))
  ```

  et le module renverra l'une des réponses suivantes :

  ```
  # 1- Si X[0, 0] n'est pas assignée à 1, cela rend le problème infaisable
  >> {"outcome": "infeasible"}

  # 2- Si X[0, 0] n'est pas assignée à 1, le problème est faisable mais on ne peut pas atteindre les mêmes scores d'optimisation
  >> {"outcome": "less_optimal",
      "optimization_scores": {"old_values": [100, 40, 70],
                              "new_values": [100, 30, 80]},
      "objective_values": [{"id": "sum_x", "old_value": 40, "new_value": 30},
                          {"id": "second_objective", "old_value": 60, "new_value": 60}...
                          ]
      }

  # 3- Si X[0, 0] n'est pas assignée à 1, on peut trouver une autre solution sans réduire les scores d'optimisation
  # Dans ce cas certaines variables auront changé (à commencer par X[0, 0]) et le module les renverra
  >> {"outcome": "as_optimal",
      "changed_variables": [{"name": x_0_0, "old_value": 1, "new_value": 0},
                           {"name": x_0_1, "old_value": 0, "new_value": 1}...]
      }

  ```

  *ExplainValueOfVar* ne permet d'étudier qu'une variable à la fois, mais *ExplainWhyNot* autorise des questions plus complexes :

  ```
  # Si dans la solution actuelle, X[0, 0] a été assignée à 1, alors les deux lignes ci-dessous ont le même effet :
  my_explanation = solver.ExplainValueOfVar(X[0, 0])
  my_explanation = solver.ExplainWhyNot(X[0, 0] != 1)

  # ExplainWhyNot permet des questions plus complexes
  my_explanation = solver.ExplainWhyNot(sum(X[i, 0] for i in list_I) == 0)
  ```
