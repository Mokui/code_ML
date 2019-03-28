import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as plt

# Factorial function
def factorial(n):
    if n <= 0:
        return 1
    else:
        return n*factorial(n-1)

# Taylor function
def function_taylor(x, max):
    fonction = np.exp
    value = 0
    plots = {}
    for k in range(0,max):
        # Using MacLaurin simplification for Taylor series
        value += x**k/factorial(k) 
        plots[k] = value
    return value, plots

def main():
    taylor, plots = function_taylor(1, 15)
    x_value = plots.keys()
    y_value = plots.values()

    plt.plot(x_value,y_value)
    plt.show()
    
if __name__ == "__main__":
    main()