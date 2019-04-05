import numpy as np
import pandas as pa
import matplotlib.pyplot as plt

def main():
    data = pa.read_csv("data/cours_bitcoin.csv",sep=';', encoding='latin-1')

    print(data) #simple display

    # data.head()
    # data.tail()

    data.stack() #Affiche en liste plutot qu'en lignes

    data.dtypes #Types de colonnes
    data.columns #liste des colonnes

    # data_tri = data[data['Année'].isin(['2016','2017'])] #Faire un tri selon l'année 
    #OR
    # data_tri = data.groupby('Année')

    data_clean = data.drop(['Variation'], axis=1) #Enleve les colonnes inutiles pour les calculs

    # data.mean() #Moyenne des valeurs

    # data2 = data.copy() 

    # debut = data.head()
    data_clean['Année'] = pa.to_datetime(data_clean.Année)

    datal = data_clean.sort_values(by=['Année'])
    #print(datal)
    datay = datal.groupby(['Année']).sum().groupby('Année').cumsum() # Calcul en fonction de l'année la somme total de déchets parmis tout les sites de fouilles
    # datan = datal[['Gravats','Déchets verts']] #Affiche uniquement les colonnes indiquées
    # datay = datan.apply(np.cumsum) #Affiche la somme ajoutée des données 
    print(datay) # affichage

    datay.plot()
    plt.show()

    # Launch me with this : %run ./bitcoin.py

if __name__ == "__main__":
    main()


