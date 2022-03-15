from src.fonction import VariablesDependantes, VariablesIndependantes, Fonction
import numpy as np
import matplotlib.pyplot as plt


def main():
    # On se crée des Variables
    x = np.linspace(0, 100, 1000)
    var_x = VariablesIndependantes(x, label="$x$")
    var_y = VariablesDependantes(np.sin(x), label=r"$\sin(x)$")

    # On peut aussi créer des VariablesIndependantes à partir de...
    var_y_2 = VariablesDependantes.from_variables_independantes(var_x, np.sin)

    # On peut s'assurer que les deux versions sont équivalentes:
    print(f"Variables indépendantes équivalentes : {var_y.egalite_totale(var_y_2)}")

    # On peut "visualiser" en string les variables, mais ça afficher toutes les valeurs internes! On peut facilement
    # voir les données en graphiques:
    plt.plot(var_x, var_y)
    plt.xlabel(var_x.label)
    plt.ylabel(var_y.label)
    plt.show()

    # On peut faire des opérations mathématiques sur les variables, comme...
    var_x_2 = var_x * 10
    var_y_2 = var_y + 1.5

    # Et les types sont les bons!
    print(type(var_x_2))
    print(type(var_y_2))

    # On peut aussi faire...
    var_y_3 = np.exp(var_x)

    # Et les types sont encore les bons!
    print(type(var_y_3))

    # À partir de Variables (ou d'intérables), on peut se construire une fonction!
    f = Fonction(var_x, var_y)

    # On peut accéder à des valeurs de cette fonction
    print(f[0])
    print(f[10])

    # On peut aussi modifier des valeurs de cette fonction (attention par contre à la cohérence entre les variables)
    f[0] = (1, 1)
    print(f[0])
    f[0] = (0, 0)

    # On peut ajouter des valeurs à cette fonction (attention encore à la cohérence)
    f.ajouter_variables((101, 10), -1)
    f.ajouter_variables((-1, -10), 0)
    print(len(f))
    plt.plot(f.x, f.y)
    plt.xlabel(f.x.label)
    plt.ylabel(f.y.label)
    plt.show()

    # On peut retirer des valeurs dde cette fonction (attention à la cohérence)
    f.enlever_variables(0)  # Ici, on retire selon l'index
    print(len(f))
    f.enlever_variables(None, 101)  # Ici, on retire selon la valeur en x
    print(len(f))
    plt.plot(f.x, f.y)
    plt.xlabel(f.x.label)
    plt.ylabel(f.y.label)
    plt.show()

    # On peut aussi ajouter des liaisons, mais ce sera pour un autre exemple.


if __name__ == '__main__':
    main()
