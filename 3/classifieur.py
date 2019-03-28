from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

digits = datasets.load_digits()

x = digits.data
y = digits.target
# print(digits.data, digits.target, digits.target_names)
split = train_test_split(x,y)

arrayClass = digits.target_names[0:2]
arrayError = []
objectVote = {}
for class1 in arrayClass:
      objectVote[class1] = 0

for data1 in split[0] : 
      for data2 in data1 : 
            val = int(data2)
            if data2 in arrayClass:
                  objectVote[data2] += 1
            else:
                  arrayError.append(data2)
                  
tp = 0
fp = len(arrayError)
for vote in objectVote:
      tp += vote

precision = tp / (tp + fp)

reg = LogisticRegression(solver='lbfgs', multi_class='auto').fit(x, y)

# print(reg.coef_)
# print(reg.intercept_)
# print(reg.predict(x))
#print(dir(reg))

tabx = []
taby = []
for i in range(0, len(split[0]), 1): 
    for data2 in split[0][i]:
        tabx.append(data2)
        taby.append(y[i])
        # print(1 / (1 + np.exp(-(data2 * reg.coef_ + reg.intercept_)))) # calcul logistique
plt.scatter(tabx, taby)
# turner()
# plt.plot(x, turner(x), color='red')
plt.show()

print("Score: ", reg.score(x, y))
print("Nombre de votes: ", objectVote)
print("Vrai positifs détectés: ", tp)
print("Faux positifs détectés: ", fp)
print("précision: ", precision)