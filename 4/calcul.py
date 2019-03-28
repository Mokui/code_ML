import numpy as np
import time

attempt_res = 1

def calcul_log(a):
    return np.log(a)

def main():
    tmps1 = time.clock()
    a = 1
    i = 1
    prec = 6
    while(True):
        print("a: ", a ," ln-->", round(calcul_log(a), prec))
        if(round(calcul_log(a), prec) < attempt_res):
            a = a + (1/i)
        if(round(calcul_log(a), prec) > attempt_res):
            a = a - (1/i)
        if(round(calcul_log(a), prec) == attempt_res):
            break
        i = i + 1
    tmps2 = time.clock()

    print("END! a -->", a)
    print("Temps d'execution = ", (tmps2-tmps1) ," secondes \n")

if __name__ == "__main__":
    main()
