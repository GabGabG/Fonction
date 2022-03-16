# Fonction
Module pour modéliser une fonction mathématique à partir de données discrètes.
## But
Le but de ce module est de modéliser des fonctions mathématiques à l'aide de données discrètes. Plus simplement, on prend des données `x` et `y` et on peut créer des objets `VariablesIndependantes` et `VariablesDependantes`. Ensuite, ces diverses variables peuvent subir des opérations mathématiques et peuvent être regrouper dans un objet `Fonction`. Cet objet permet notamment de lier les variables par une liaison quelconque. Ces liaisons sont des objets `_Liaison`, qui peuvent être une régression ou une interpolation. Ainsi, il est possible de lier les données de différentes manières.

## Contenu
### Aspect mathématique et logique
Le code est séparé en différentes classes. Il y a premièrememt les classes `_Variables`, `VariablesIndependantes` et `VariablesDependantes`. Ces deux dernières classes héritent de `_Variables` et *modélisent* ce qu'on appelle une variable en mathématique. Cette variable peut être indépendante (souvent simplement appellée *x*), mais elle peut aussi être dépendante (souvent simplement appellée *y*). On peut effectuer une panoplie d'opérations mathématiques, autant des opérations *builtins* que des opérations *NumPy* (l'objet `_Variables` est considéré comme un *array-like*).

La classe `Fonction` est aussi présente pour logiquement regrouper des variables indépendantes et des variables dépendantes. Ces variables, au sein d'un objet `Fonction` peuvent aussi être mathématiquement liées avec un objet `_Liaison`. Cette classe, présentée un peu plus loin, permet de *lier* les variables entre elles avec, par exemple, une interpolation cubique ou une régression exponentielle. Éventuellement, il sera question de pouvoir afficher un objet `Fonction` grâce à *Matplotlib* et les classes `Courbe` et `Courbes` (présentées plus loin).

La classe `_Liaison` est à la base de ce qu'on appelle *lier* des variables indépendantes et dépendantes. Elle est *protégée* (notez `_` devant le nom), car elle ne devrait jamais être utilisée comme elle est; on devrait toujours définir une classe qui en hérite et qui a une logique qui a du sens. Ensuite, il existe la classe `LiaisonMixte`. Cette classe permet de créer un mixte d'objets `_Liaisons`. Cela peut être utile lorsqu'on a, par exemple, besoin de faire une régression exponentielle, suivie d'une régression gaussienne dans un jeu de données. Il est important de savoir que les liaisons utilisées peuvent permettre les discontinuités. Les discontinuités sont les espaces vides entre les différentes liaisons. Lorsqu'on permet les discontinuités, on les permet pour toutes les largeurs. Par contre, lorsqu'elles ne sont pas permises, on peut spécifier une taille, nommée `epsilon_continuite` qui nous indique quelle est la tolérance maximale permise entre les liaisons. Finalement, il existe des classes définies pour modéliser des régressions polynomiales et générales, ainsi que des interpolations (autant linéaires que quadratiques ou cubiques).

### Aspect visualisation
**PARTIE POUVANT ENCORE CHANGER (beaucoup)**
Il existe les classes `Courbe` et `Courbes` représentant une courbe unique (sur une figure) ou plusieurs courbes (sur une même figure). Cette partie n'est pas terminée, notamment car elle `Fonction` devrait l'utiliser.
