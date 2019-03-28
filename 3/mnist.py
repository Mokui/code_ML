from sklearn import datasets

digits = datasets.load_digits()
iris = datasets.load_iris() # Données sur les fleurs

for dz in digits.data:
    print(dz)

'''
    On a donc ->
    data : Données au format vectoriel (= xi)
    target : total des classes par objet (= yi)
    target_names : Liste des classes de données 
    images : liste des images en 8x8
    DESCR : Description du contenu MNIST
'''



