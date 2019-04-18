import time
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split  


def load_data():
    lfw = datasets.fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    return lfw.data, lfw.target


def show_graph(X, y_pred):
    # Time calcul
    t1_start = time.perf_counter()
    t2_start = time.process_time()

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
    for key, value in dictio.items() :
        # On récupère les données des tuples à afficher et on leur affecte une couleur différente à chaque fois
        plt.scatter([v[0] for v in value], [v[1] for v in value], s=30, label=f'Classifier {key}')

    plt.legend()

    # End Time calcul
    t1_stop = time.perf_counter()
    t2_stop = time.process_time()
    print("Elapsed time: %.1f [sec]" % ((t1_stop-t1_start)))
    print("CPU process time: %.1f [sec]" % ((t2_stop-t2_start)))

    plt.show()


def svm_func(X_train, X_test, y_train):
    svclassifier = SVC(kernel='linear').fit(X_train, y_train)
    return svclassifier.predict(X_test)


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    y_pred = svm_func(X_train, X_test, y_train)
    
    show_graph(X_test, y_pred)

if __name__ == "__main__":
    main()