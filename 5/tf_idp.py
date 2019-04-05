from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pa

def main():

    with open('le_sorceleur.txt', 'r') as file:
    doc = file.read()

    # print(doc)

    corpus = [doc]

    vectorizer = TfidfVectorizer()
    valeurs = vectorizer.fit_transform(corpus)
    liste = vectorizer.get_feature_names()

    # print(liste)
    # print(valeurs.data)

    # Affecte chaque proba d'apparition à un mot
    dico = {}
    for i in range(0, len(valeurs.data), 1):
    dico[liste[i]] = valeurs.data[i]

    value = sorted(dico, key=lambda dico: dico[1], reverse=True)
    print(pa.DataFrame([value]))

if __name__ == "__main__":
    main()