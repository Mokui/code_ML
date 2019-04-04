# coding: utf-8
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime

def format_date(myDate) :
    date_format_date = "%Y-%m-%d"
    f_date = datetime.strptime("2019-03-16", date_format_date)
    l_date = datetime.strptime(myDate, date_format_date)
    delta = l_date - f_date
    return delta.days

# May need to reformat_date this code or use sklearn, quite complicated in 10minutes.
def main():
    data = [[format_date("2019-03-16"), 81682.0], [format_date("2019-03-18"), 81720.0], [format_date("2019-03-20"), 81760.0], 
    [format_date("2019-03-24"), 81826.0], [format_date("2019-03-25"), 81844.0], [format_date("2019-03-26"), 81864.0], [format_date("2019-03-27"),81881.0], 
    [format_date("2019-03-28"),81900.0], [format_date("2019-03-30"),81933.0], [format_date("2019-04-03"), 82003.0]]
    
    print(data)

    x = [[elem[0]] for elem in data]
    y = [elem[1] for elem in data]
    print(x)
    print(y)

    reg = LinearRegression().fit(x, y)

    arr = np.arange(10)
    ys = [i+arr+(i*arr)**2 for i in range(10)]

    colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    plt.scatter(x, y, 30, colors) #Here come the rainbow

    y_pred = reg.predict(x) 
    next_y = (reg.predict([[20]]))
    
    plt.plot(x, y_pred, 'c-', lw=2)

    plt.legend(next_y) # affiche le prochain Y
    #plt.plot([next_y, next_y], [0.0, 90000], 'r-', lw=2) #Wanna plot a single straight vertical line but it erase my previous plot for no reason
    
    plt.show() 

if __name__ == "__main__":
    main()