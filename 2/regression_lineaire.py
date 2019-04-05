import numpy as np
from sklearn.linear_model import LinearRegression

def main(): 
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    # y = 1 * x_0 + 2 * x_1 + 3
    print(X) 

    y = np.dot(X, np.array([1, 2])) + 3
    reg = LinearRegression().fit(X, y)

    print("Score de corrélation ", reg.score(X, y))

    print("coef trajectoire courbe", reg.coef_)

    print("intercept", reg.intercept_)

    print("predict", reg.predict(np.array([[3, 5]])))

    print("---------------------------------------------")

    """ y = np.dot(X, np.array([1, 2])) + 1
    reg = LinearRegression().fit(X, y)

    print("Score de corrélation ", reg.score(X, y))

    print("coef trajectoire courbe ", reg.coef_)

    print("intercept", reg.intercept_)

    print("predict", reg.predict(np.array([[3, 5]])))

    print("---------------------------------------------") """

if __name__ == "__main__":
    main()