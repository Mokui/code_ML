# coding: utf-8
from matplotlib import pyplot as plt
import numpy as np


def calcul_hypothese(teta0, teta1, x):
    return teta0 + (teta1 * x)


def fonction_cout(teta0, teta1, data):
    return (1/(2*len(data))) * sum([
        ((calcul_hypothese(teta0, teta1, x)) - y) ** 2 for x, y in data
    ])


def gradient_descent(alpha, teta0, teta1, data):
    return {
        "teta0": teta0 - (alpha/len(data)) * sum([
            calcul_hypothese(teta0, teta1, x) - y for x, y in data
        ]),
        "teta1": teta1 - (alpha/len(data)) * sum([
            (calcul_hypothese(teta0, teta1, x) - y) * x for x, y in data
        ]),
    }


def set_alpha(n):
    return 1 / n


def ruler_while(pre, array):
    print(array)
    if round(array[-1]["teta0"], pre) == round(array[-2]["teta0"], pre) and round(array[-1]["teta1"], pre) == round(array[-2]["teta1"], pre):
        return False
    return True

def format(date) :
    return np.datetime64(date)

# MAy need to reformat this code or use sklearn, quite complicated in 10minutes.
def main():
    data = [[format("2019-03-16"), 81682.0], [format("2019-03-18"), 81720.0], [format("2019-03-20"), 81760.0], 
    [format("2019-03-24"), 81826.0], [format("2019-03-25"), 81844.0], [format("2019-03-26"), 81864.0], [format("2019-03-27"),81881.0], 
    [format("2019-03-28"),81900.0], [format("2019-03-30"),81933.0], [format("2019-04-03"), 82003.0]]

    teta0 = 1
    teta1 = 2
    alpha = 1
    n = 0
    array = [
        {"teta0": 1, "teta1": 2}
    ]
    while True:
        fonc_cout = fonction_cout(teta0, teta1, data)
        grad_desc = gradient_descent(alpha, teta0, teta1, data)
        print(f"cout: {fonc_cout}, gradient: {grad_desc}")
        array.append(grad_desc)
        if not ruler_while(30, array):
            break
        print(alpha)
        n += 1
        alpha = set_alpha(n)
    print(f"finish !!! cout: {fonc_cout}, gradient: {grad_desc}")

    x = [elem[0] for elem in data]
    y = [elem[1] for elem in data]
    print(x)
    print(y)
    # x, y
    plt.subplot(211)
    plt.scatter(x, y)
    x = np.arange(0, 8, 0.1)
    y = [calcul_hypothese(teta0, teta1, elem) for elem in x]
    plt.plot(x, y)
    # x, residu
    # plt.subplot(212)
    # plt.scatter(x, y)
    # plt.plot(x, )
    # add a polar subplot
    plt.show()

if __name__ == "__main__":
    main()