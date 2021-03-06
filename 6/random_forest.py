from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pydot
import time

def main():
    # Time calcul
    t1_start = time.perf_counter()
    t2_start = time.process_time()

    # Else charge this Mnist data
    iris = load_iris()

    # print(iris.DESCR)
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

    # Train the model on training data
    rf.fit(iris.data, iris.target)

    # Pull out one tree from the forest
    tree = rf.estimators_[5]

    # Export the image to a dot file
    export_graphviz(tree, out_file = 'tree.dot', feature_names = iris.feature_names, rounded = True, precision = 1)

    # Use dot file to create a graph
    (graph, ) = pydot.graph_from_dot_file('tree.dot')

    # Write graph to a png file
    graph.write_png('tree.png')

    # End Time calcul
    t1_stop = time.perf_counter()
    t2_stop = time.process_time()
    print("Elapsed time: %.1f [sec]" % ((t1_stop-t1_start)))
    print("CPU process time: %.1f [sec]" % ((t2_stop-t2_start)))

if __name__ == "__main__":
    main()