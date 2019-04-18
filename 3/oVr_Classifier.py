from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from numpy.random import rand
import matplotlib.pyplot as plt
import numpy as np
import time


def main():
    # Time calcul
    t1_start = time.perf_counter()
    t2_start = time.process_time()

    # This is an example of a OvO classifier use from sklearn
    # Got it from https://scikit-learn.org/stable/modules/multiclass.html#ovo-classification 
    iris = datasets.load_iris() 
    X, y = iris.data, iris.target

    fitter = OneVsRestClassifier(LinearSVC(random_state=0, max_iter=50000)).fit(X, y)

    y_pred = fitter.predict(X)
    # print(y_pred)
    # print(X)

    dictio = {}
    tab = []

    # Crée un tableau avec les classifieurs 0,1,2 
    for i in range(0, len(y_pred)):
        if y_pred[i] not in tab:
            tab.append(y_pred[i])
        else: 
            continue
    
    # Crée un dictionnaire avec les classifieurs 0,1,2 (par exemple)
    dictio = {str(key): [] for key in tab}
    
    # Ajoute les valeurs X correspondantes à leurs classifieurs
    for i in range(0, len(y_pred)):
        dictio[str(y_pred[i])].append(X[i])

    # Scatter + couleurs
    colors = ['r','c','y']
    selector = 0
    for key, value in dictio.items() :
        # On récupère les données des tuples à afficher et on leur affecte une couleur différente à chaque fois
        plt.scatter([v[0] for v in value], [v[1] for v in value], color=colors[selector], label=f'Classifier {key}')
        selector += 1

    plt.legend()
    plt.show()

    # End Time calcul
    t1_stop = time.perf_counter()
    t2_stop = time.process_time()
    print("Elapsed time: %.1f [sec]" % ((t1_stop-t1_start)))
    print("CPU process time: %.1f [sec]" % ((t2_stop-t2_start)))

if __name__ == "__main__":
    main()
