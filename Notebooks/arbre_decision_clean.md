# Importation des librairies


```python
import numpy as np
import pandas as pd
from math import log
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from IPython.display import Image  
from sklearn import tree
import pydotplus
import matplotlib.pyplot as plt
import seaborn as sns
import random 
from sklearn.metrics import accuracy_score
import six
from sklearn.model_selection import train_test_split
```


```python
#import warnings
#warnings.filterwarnings("ignore")
```

# Création du jeu de données


```python
data = {'age' : ['<=30','<=30','31-40','>40','>40','>40','31-40','<=30','<=30','>40','<=30','31-40','31-40','>40'],
        'revenu' : ['eleve','eleve','eleve','moyen','faible','faible','faible','moyen','faible','moyen','moyen','moyen','eleve','moyen'],
        'etudiant': ['non','non','non','non','oui','oui','oui','non','oui','oui','oui','non','oui','non'],
        'credit' : ['bon','excellent','bon','bon','bon','excellent','excellent','bon','bon','bon','excellent','excellent','bon','excellent'],
        'achat' : [0,0,1,1,1,0,1,0,1,1,1,1,1,0]}
```


```python
df = pd.DataFrame(data)
display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>revenu</th>
      <th>etudiant</th>
      <th>credit</th>
      <th>achat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>&lt;=30</td>
      <td>eleve</td>
      <td>non</td>
      <td>bon</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>&lt;=30</td>
      <td>eleve</td>
      <td>non</td>
      <td>excellent</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>31-40</td>
      <td>eleve</td>
      <td>non</td>
      <td>bon</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>&gt;40</td>
      <td>moyen</td>
      <td>non</td>
      <td>bon</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>&gt;40</td>
      <td>faible</td>
      <td>oui</td>
      <td>bon</td>
      <td>1</td>
    </tr>
    <tr>
      <td>5</td>
      <td>&gt;40</td>
      <td>faible</td>
      <td>oui</td>
      <td>excellent</td>
      <td>0</td>
    </tr>
    <tr>
      <td>6</td>
      <td>31-40</td>
      <td>faible</td>
      <td>oui</td>
      <td>excellent</td>
      <td>1</td>
    </tr>
    <tr>
      <td>7</td>
      <td>&lt;=30</td>
      <td>moyen</td>
      <td>non</td>
      <td>bon</td>
      <td>0</td>
    </tr>
    <tr>
      <td>8</td>
      <td>&lt;=30</td>
      <td>faible</td>
      <td>oui</td>
      <td>bon</td>
      <td>1</td>
    </tr>
    <tr>
      <td>9</td>
      <td>&gt;40</td>
      <td>moyen</td>
      <td>oui</td>
      <td>bon</td>
      <td>1</td>
    </tr>
    <tr>
      <td>10</td>
      <td>&lt;=30</td>
      <td>moyen</td>
      <td>oui</td>
      <td>excellent</td>
      <td>1</td>
    </tr>
    <tr>
      <td>11</td>
      <td>31-40</td>
      <td>moyen</td>
      <td>non</td>
      <td>excellent</td>
      <td>1</td>
    </tr>
    <tr>
      <td>12</td>
      <td>31-40</td>
      <td>eleve</td>
      <td>oui</td>
      <td>bon</td>
      <td>1</td>
    </tr>
    <tr>
      <td>13</td>
      <td>&gt;40</td>
      <td>moyen</td>
      <td>non</td>
      <td>excellent</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
# On isole la variable dépendantes et les variables indépendantes
y = df.achat
X = df.drop('achat', axis=1)
```


```python
display(y)
```


    0     0
    1     0
    2     1
    3     1
    4     1
    5     0
    6     1
    7     0
    8     1
    9     1
    10    1
    11    1
    12    1
    13    0
    Name: achat, dtype: int64



```python
display(X)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>revenu</th>
      <th>etudiant</th>
      <th>credit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>&lt;=30</td>
      <td>eleve</td>
      <td>non</td>
      <td>bon</td>
    </tr>
    <tr>
      <td>1</td>
      <td>&lt;=30</td>
      <td>eleve</td>
      <td>non</td>
      <td>excellent</td>
    </tr>
    <tr>
      <td>2</td>
      <td>31-40</td>
      <td>eleve</td>
      <td>non</td>
      <td>bon</td>
    </tr>
    <tr>
      <td>3</td>
      <td>&gt;40</td>
      <td>moyen</td>
      <td>non</td>
      <td>bon</td>
    </tr>
    <tr>
      <td>4</td>
      <td>&gt;40</td>
      <td>faible</td>
      <td>oui</td>
      <td>bon</td>
    </tr>
    <tr>
      <td>5</td>
      <td>&gt;40</td>
      <td>faible</td>
      <td>oui</td>
      <td>excellent</td>
    </tr>
    <tr>
      <td>6</td>
      <td>31-40</td>
      <td>faible</td>
      <td>oui</td>
      <td>excellent</td>
    </tr>
    <tr>
      <td>7</td>
      <td>&lt;=30</td>
      <td>moyen</td>
      <td>non</td>
      <td>bon</td>
    </tr>
    <tr>
      <td>8</td>
      <td>&lt;=30</td>
      <td>faible</td>
      <td>oui</td>
      <td>bon</td>
    </tr>
    <tr>
      <td>9</td>
      <td>&gt;40</td>
      <td>moyen</td>
      <td>oui</td>
      <td>bon</td>
    </tr>
    <tr>
      <td>10</td>
      <td>&lt;=30</td>
      <td>moyen</td>
      <td>oui</td>
      <td>excellent</td>
    </tr>
    <tr>
      <td>11</td>
      <td>31-40</td>
      <td>moyen</td>
      <td>non</td>
      <td>excellent</td>
    </tr>
    <tr>
      <td>12</td>
      <td>31-40</td>
      <td>eleve</td>
      <td>oui</td>
      <td>bon</td>
    </tr>
    <tr>
      <td>13</td>
      <td>&gt;40</td>
      <td>moyen</td>
      <td>non</td>
      <td>excellent</td>
    </tr>
  </tbody>
</table>
</div>



```python
# On va utiliser l'algorithme Cart donc on doit avoir que des embranchements binaires 
# Certaines variables ont plus que 2 modalités (niveau de granularité plus fin)
X = pd.get_dummies(data=X, drop_first=True)
```


```python
display(X)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age_&lt;=30</th>
      <th>age_&gt;40</th>
      <th>revenu_faible</th>
      <th>revenu_moyen</th>
      <th>etudiant_oui</th>
      <th>credit_excellent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



```python
def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax
```


```python
render_mpl_table(X, header_columns=0, col_width=2)
plt.tight_layout()
plt.show()
#plt.savefig('keras_results.svg')
```


![png](arbre_decision_clean_files/arbre_decision_clean_12_0.png)


# Algorithme arbre de décision

## Fonction meilleur_split


```python
def meilleur_split(X, y, indice='gini'):
    #liste_gini_all_features = [] # Pour la visualisation graphique ci-après
    liste_finale = [] # Permet de stocker les informations du meilleur feature (indice d'impureté pondéré le plus faible)
    z = 0 # On instancie un compteur 
    proportion_noeud_pere = y.loc[X.index].sum() / y.loc[X.index].shape[0] # On peut faire loc sur des index car ce sont les labels des rows
                                                                           # calcul somme des true pour tout le df
    for i in X.columns: # On itère sur chaque feature dans notre dataframe
        for j in X[i].unique(): # On itère sur chaque modalité unique comprise dans chaque feature 
            true = X[X[i] > j].index # On pose les questions pour trouver les chemins true et false
            false = X[X[i] <= j].index # On utilise les index pour quand on aura un subset du dataframe (Surtout pour les variables continues)
            
            #true_df = X.loc[true].index
            #false_df = X[false].index
            
            if len(y[true].index) == 0 or len(y[false].index) == 0: # Pour se prémunir d'un dénominateur nulle quand il n'y a aucune observation qui respecte la condition
                continue # Permet à la boucle for de sauter à l'itération suivante 
            
            proportion_true = y.loc[true].sum() / y.loc[true].shape[0] # Proportion de 1 pour true
            proportion_false = y.loc[false].sum() / y.loc[false].shape[0] # Proportion de 1 pour false
        
            if indice == 'gini': # Version si l'indice d'impureté est gini
                
                gini_feature_true = 1 - (y[true].sum()/len(y[true]))**2 - (1 - (y[true].sum()/len(y[true])))**2 # calcul de gini pour true
                gini_feature_false = 1 - (y[false].sum()/len(y[false]))**2 - (1 - (y[false].sum()/len(y[false])))**2 # calcul de gini pour false
        
                impurete_feature = len(y[true])/len(y[true] + y[false])*gini_feature_true + len(y[false])/len(y[true] + y[false])*gini_feature_false # calcul de gini pondéré
        
                impurete_noeud_pere = 1 - (proportion_noeud_pere**2) - (1 - proportion_noeud_pere)**2 # calcul indice de gini du pere 
            
                gain_info = impurete_noeud_pere - impurete_feature # calcul du gain informationnel
            
            
            if indice == 'entropie': # version si l'indice d'impureté est entropie
                if proportion_true == 0 or proportion_true == 1:
                    entropie_feature_true = 0
                    
                elif proportion_true != 0 or proportion_true != 1:
                    entropie_feature_true = -(proportion_true*(log(proportion_true)/log(2)) + (1-proportion_true)*(log(1-proportion_true)/log(2)))
                
                if proportion_false == 0 or proportion_false == 1:
                    entropie_feature_false = 0
                    
                elif proportion_false != 0 or proportion_false != 1:
                    entropie_feature_false = -(proportion_false*(log(proportion_false)/log(2)) + (1-proportion_false)*(log(1-proportion_false)/log(2)))
                
                impurete_feature = len(y[true])/len(y[true] + y[false])*entropie_feature_true + len(y[false])/len(y[true] + y[false])*entropie_feature_false
                
                impurete_noeud_pere = -(proportion_noeud_pere*(log(proportion_noeud_pere)/log(2)) + (1-proportion_noeud_pere)*(log(1-proportion_noeud_pere)/log(2)))

                gain_info = impurete_noeud_pere - impurete_feature
                
            liste_temp = [i, j, gain_info, impurete_feature, proportion_true, proportion_false, true, false] # On stock les informations de manière temporaire
        
            #liste_gini_all_features.append(liste_temp) # Pour la visualisation graphique ci-après
            
            if z == 0: 
                liste_finale = liste_temp # Contient les informations du feature évalué lors de la 1ère itération
                z += 1 # On incrémente la valeur de z de 1
            
            elif z > 0:
                if impurete_feature < liste_finale[3]: # Si l'indice d'impureté pondéré < celui dans liste_finale
                    liste_finale = liste_temp # On met à jour les informations de liste_finale 
                
            true_path = X.loc[liste_finale[6]].drop(liste_finale[0], axis=1) # Mise à jour du dataframe avec le chemin gauche
            false_path = X.loc[liste_finale[7]].drop(liste_finale[0], axis=1) # Mise à jour du dataframe avec le chemin droit
            
    return true_path, false_path, liste_finale # On retourne le chemin vrai, le chemin faux et les informations de liste_finale
    #return liste_gini_all_features
```

### Visualisation graphique


```python
# Visualiser les indices de gini pour tous les features
df_gini_all_features = pd.DataFrame(data=meilleur_split(X, y))
display(df_gini_all_features)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>age_&lt;=30</td>
      <td>0</td>
      <td>0.065533</td>
      <td>0.393651</td>
      <td>0.400000</td>
      <td>0.777778</td>
      <td>Int64Index([0, 1, 7, 8, 10], dtype='int64')</td>
      <td>Int64Index([2, 3, 4, 5, 6, 9, 11, 12, 13], dty...</td>
    </tr>
    <tr>
      <td>1</td>
      <td>age_&gt;40</td>
      <td>0</td>
      <td>0.002041</td>
      <td>0.457143</td>
      <td>0.600000</td>
      <td>0.666667</td>
      <td>Int64Index([3, 4, 5, 9, 13], dtype='int64')</td>
      <td>Int64Index([0, 1, 2, 6, 7, 8, 10, 11, 12], dty...</td>
    </tr>
    <tr>
      <td>2</td>
      <td>revenu_faible</td>
      <td>0</td>
      <td>0.009184</td>
      <td>0.450000</td>
      <td>0.750000</td>
      <td>0.600000</td>
      <td>Int64Index([4, 5, 6, 8], dtype='int64')</td>
      <td>Int64Index([0, 1, 2, 3, 7, 9, 10, 11, 12, 13],...</td>
    </tr>
    <tr>
      <td>3</td>
      <td>revenu_moyen</td>
      <td>0</td>
      <td>0.000850</td>
      <td>0.458333</td>
      <td>0.666667</td>
      <td>0.625000</td>
      <td>Int64Index([3, 7, 9, 10, 11, 13], dtype='int64')</td>
      <td>Int64Index([0, 1, 2, 4, 5, 6, 8, 12], dtype='i...</td>
    </tr>
    <tr>
      <td>4</td>
      <td>etudiant_oui</td>
      <td>0</td>
      <td>0.091837</td>
      <td>0.367347</td>
      <td>0.857143</td>
      <td>0.428571</td>
      <td>Int64Index([4, 5, 6, 8, 9, 10, 12], dtype='int...</td>
      <td>Int64Index([0, 1, 2, 3, 7, 11, 13], dtype='int...</td>
    </tr>
    <tr>
      <td>5</td>
      <td>credit_excellent</td>
      <td>0</td>
      <td>0.030612</td>
      <td>0.428571</td>
      <td>0.500000</td>
      <td>0.750000</td>
      <td>Int64Index([1, 5, 6, 10, 11, 13], dtype='int64')</td>
      <td>Int64Index([0, 2, 3, 4, 7, 8, 9, 12], dtype='i...</td>
    </tr>
  </tbody>
</table>
</div>



```python
# On garde juste les features et les indices de gini pondérés des features
df_gini_all_features = df_gini_all_features.loc[:, [0,3]]
df_gini_all_features.columns = ['Features', 'Gini index']
```


```python
df_gini_all_features.sort_values(by='Gini index', inplace=True)
```


```python
sns.barplot(x='Features', y='Gini index', data=df_gini_all_features)
plt.xticks(rotation=45)
plt.ylabel('Gini pondéré')
plt.tight_layout()
plt.show()
#plt.savefig('indice_gini.svg')
```


![png](arbre_decision_clean_files/arbre_decision_clean_20_0.png)


## Fonction construction_arbre


```python
def construction_arbre(X, y, max_depth=3, gain_info_min = 0.01, indice='gini'): # Hyper paramètres pour limiter la croissance de l'arbre
    nb_elements = 2**((max_depth+1)) # Nombre d'éléments dans la liste indexé à 0
    binary_heap = [0] * nb_elements # Creation d'une liste vierge    
    
    for i in range(1, 2**max_depth): # Pour ne pas construire les enfants du dernier etage vs nb elements auparavant
        if i == 1:
            binary_heap[1] = X.index
            split1 = meilleur_split(X, y, indice)
            binary_heap[2] = list(split1[2][s] for s in [0, 1, 2, 3, 4, 6])
            binary_heap[3] = list(split1[2][s] for s in [0, 1, 2, 3, 5, 7])
            
        elif i != 1:
            if binary_heap[i] != 0: # Pour prévenir les cas ou il n'y a pas d'enfants
                gain_info = binary_heap[i][2]
                proportion_true_false = binary_heap[i][-2]
            
                if (gain_info > gain_info_min) & (proportion_true_false != 0) & (proportion_true_false != 1):
                
                    index = binary_heap[i][-1]
                    df = X.loc[index]
                    split = meilleur_split(df, y, indice)
            
                    binary_heap[2*i] = list(split[2][s] for s in [0, 1, 2, 3, 4, 6])
                    binary_heap[(2*i)+1] = list(split[2][s] for s in [0, 1, 2, 3, 5, 7])
                      
    return binary_heap         
                
# Prendre un compte les enfants dans le for loop car on appelle binary_heap[i]
```


```python
binary_heap = construction_arbre(X, y, max_depth=3, gain_info_min = 0.01, indice='gini')
```


```python
binary_heap
```




    [0,
     RangeIndex(start=0, stop=14, step=1),
     ['etudiant_oui',
      0,
      0.09183673469387743,
      0.3673469387755103,
      0.8571428571428571,
      Int64Index([4, 5, 6, 8, 9, 10, 12], dtype='int64')],
     ['etudiant_oui',
      0,
      0.09183673469387743,
      0.3673469387755103,
      0.42857142857142855,
      Int64Index([0, 1, 2, 3, 7, 11, 13], dtype='int64')],
     ['age_>40',
      0,
      0.054421768707483054,
      0.19047619047619047,
      0.6666666666666666,
      Int64Index([4, 5, 9], dtype='int64')],
     ['age_>40',
      0,
      0.054421768707483054,
      0.19047619047619047,
      1.0,
      Int64Index([6, 8, 10, 12], dtype='int64')],
     ['age_<=30',
      0,
      0.27551020408163274,
      0.21428571428571427,
      0.0,
      Int64Index([0, 1, 7], dtype='int64')],
     ['age_<=30',
      0,
      0.27551020408163274,
      0.21428571428571427,
      0.75,
      Int64Index([2, 3, 11, 13], dtype='int64')],
     ['credit_excellent',
      0,
      0.4444444444444444,
      0.0,
      0.0,
      Int64Index([5], dtype='int64')],
     ['credit_excellent',
      0,
      0.4444444444444444,
      0.0,
      1.0,
      Int64Index([4, 9], dtype='int64')],
     0,
     0,
     0,
     0,
     ['age_>40', 0, 0.125, 0.25, 0.5, Int64Index([3, 13], dtype='int64')],
     ['age_>40', 0, 0.125, 0.25, 1.0, Int64Index([2, 11], dtype='int64')]]



## Fonction prediction


```python
def prediction(df, binary_heap):
    
    i = 2 # C'est le début du heap car i=1 contient tout le dataframe
    proportion = 0 # Le mettre en dehors car dans la boucle while, c'est une variable locale
    
    while i < len(binary_heap): # Ne pas dépasser le nombre de feuilles terminales 
        if binary_heap[i] != 0: # Éviter les cas où il n'y a pas d'enfants
            if df[binary_heap[i][0]] > binary_heap[i][1]: # On repose la question (chemin vrai)
                proportion = binary_heap[i][-2] # Contient l'information de la proportion de 1 dans true
                i = 2*i # Enfant de true se trouve à gauche
            elif df[binary_heap[i][0]] <= binary_heap[i][1]: # On repose la question (chemin faux)
                proportion = binary_heap[i+1][-2] # Contient l'information de la proportion de 1 dans false
                i = 2*(i+1) # Enfant de false se trouve à droite
                
        elif binary_heap[i] == 0: # Pour empêcher de rerentrer dans la boucle while car on ne met pas à jour le i
            break # La boucle s'arrête dès qu'on tombe sur une feuille terminale
    
    return proportion # On retourne la proportion de 1 dans true ou false
```

# Test de rapidité

## Dataframe généré aléatoirement 
Pour tester la rapidité de prédiction de notre algorithme sur un gros jeu de données


```python
xpred = {'age_<=30':np.random.randint(2, size=1000000), 'age_>40':np.random.randint(2, size=1000000), 'revenu_faible':np.random.randint(2, size=1000000), 'revenu_moyen':np.random.randint(2, size=1000000), 'etudiant_oui':np.random.randint(2, size=1000000), 'credit_excellent':np.random.randint(2, size=1000000), 'achat':np.random.randint(2, size=1000000)}
xpred_df = pd.DataFrame(xpred)
display(xpred_df.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age_&lt;=30</th>
      <th>age_&gt;40</th>
      <th>revenu_faible</th>
      <th>revenu_moyen</th>
      <th>etudiant_oui</th>
      <th>credit_excellent</th>
      <th>achat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



```python
xpred_df.shape
```




    (1000000, 7)




```python
X = xpred_df.drop('achat', axis=1)
y = xpred_df.achat
```


```python
binary_heap = construction_arbre(X, y, max_depth=3, gain_info_min = 0.01, indice='gini')
```

`Prédiction avec une for loop`


```python
for i in xpred_df.index:
    xpred_df['prediction'] = prediction(xpred_df.loc[i], binary_heap)
```

`Prédiction avec une list comprehension`


```python
# Code avec une list comprehension (compilé en c++)
xpred_df['prediction'] = [prediction(xpred_df.loc[i], binary_heap) for i in xpred_df.index]
```

`Prédiction avec apply`


```python
# Code avec apply (méthode de pandas)
xpred_df['prediction'] = xpred_df.apply(lambda df: prediction(df, binary_heap), axis=1)
```

`Dataframe résumant les méthodes (complexité différentes) et le temps d'exécution`


```python
dict_results = {'Méthodes':['for loop', 'list comprehension', 'apply'], 'Temps d\'exécution':['38min 38s', '2min 7s', '14.8s']}

df_results = pd.DataFrame(dict_results)

render_mpl_table(df_results, col_width=4, row_height=1)
plt.tight_layout()
plt.show()
```


![png](arbre_decision_clean_files/arbre_decision_clean_40_0.png)


## Arbre de décision de sklearn
En utilisant les mêmes hyper paramètres que notre arbre de décision


```python
xpred = {'age_<=30':np.random.randint(2, size=1000000), 'age_>40':np.random.randint(2, size=1000000), 'revenu_faible':np.random.randint(2, size=1000000), 'revenu_moyen':np.random.randint(2, size=1000000), 'etudiant_oui':np.random.randint(2, size=1000000), 'credit_excellent':np.random.randint(2, size=1000000),'achat':np.random.randint(2, size=1000000)}
xpred_df = pd.DataFrame(xpred)
display(xpred_df.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age_&lt;=30</th>
      <th>age_&gt;40</th>
      <th>revenu_faible</th>
      <th>revenu_moyen</th>
      <th>etudiant_oui</th>
      <th>credit_excellent</th>
      <th>achat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
X = xpred_df.drop('achat', axis=1)
y = xpred_df.achat
```


```python
clf = DecisionTreeClassifier(max_depth=3, min_impurity_split=0.01) # On utilise les mêmes hyper paramètres
```


```python
clf.fit(X, y)
```

    /Users/florentfettu/opt/anaconda3/lib/python3.7/site-packages/sklearn/tree/tree.py:297: DeprecationWarning: The min_impurity_split parameter is deprecated. Its default value will change from 1e-7 to 0 in version 0.23, and it will be removed in 0.25. Use the min_impurity_decrease parameter instead.
      DeprecationWarning)





    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=0.01,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort=False,
                           random_state=None, splitter='best')




```python
clf.predict(X)
```




    array([0, 1, 0, ..., 1, 1, 1])



# Test de performance

## Prédiction sur le jeu de données « Iris »
Pour tester la performance de prédiction de notre algorithme


```python
iris_df = pd.read_csv('iris.csv')
display(iris_df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <td>1</td>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <td>2</td>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>145</td>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
      <td>virginica</td>
    </tr>
    <tr>
      <td>146</td>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
      <td>virginica</td>
    </tr>
    <tr>
      <td>147</td>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
      <td>virginica</td>
    </tr>
    <tr>
      <td>148</td>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>virginica</td>
    </tr>
    <tr>
      <td>149</td>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>virginica</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 5 columns</p>
</div>



```python
new_iris_df = iris_df[iris_df.species != 'setosa'] # On supprime une classe pour avoir un y binaire
display(new_iris_df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>50</td>
      <td>7.0</td>
      <td>3.2</td>
      <td>4.7</td>
      <td>1.4</td>
      <td>versicolor</td>
    </tr>
    <tr>
      <td>51</td>
      <td>6.4</td>
      <td>3.2</td>
      <td>4.5</td>
      <td>1.5</td>
      <td>versicolor</td>
    </tr>
    <tr>
      <td>52</td>
      <td>6.9</td>
      <td>3.1</td>
      <td>4.9</td>
      <td>1.5</td>
      <td>versicolor</td>
    </tr>
    <tr>
      <td>53</td>
      <td>5.5</td>
      <td>2.3</td>
      <td>4.0</td>
      <td>1.3</td>
      <td>versicolor</td>
    </tr>
    <tr>
      <td>54</td>
      <td>6.5</td>
      <td>2.8</td>
      <td>4.6</td>
      <td>1.5</td>
      <td>versicolor</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>145</td>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
      <td>virginica</td>
    </tr>
    <tr>
      <td>146</td>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
      <td>virginica</td>
    </tr>
    <tr>
      <td>147</td>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
      <td>virginica</td>
    </tr>
    <tr>
      <td>148</td>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>virginica</td>
    </tr>
    <tr>
      <td>149</td>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>virginica</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 5 columns</p>
</div>



```python
#display(new_iris_df.describe())
#bins = [0, 2, 4, 6, np.inf]
#names = ['<2', '2-4', '4-6', '6+']

#for i in new_iris_df.columns:
#    if i != 'species':
#        new_iris_df[i] = pd.cut(new_iris_df[i], bins, labels=names) # Permet de créer des ranges
```


```python
mapping = {'versicolor':0, 'virginica':1} # Convertir y en numérique

new_iris_df['species'] = new_iris_df.species.map(mapping)
```

    /Users/florentfettu/opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      This is separate from the ipykernel package so we can avoid doing imports until



```python
#new_iris_df = pd.get_dummies(new_iris_df)
```


```python
display(new_iris_df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>50</td>
      <td>7.0</td>
      <td>3.2</td>
      <td>4.7</td>
      <td>1.4</td>
      <td>0</td>
    </tr>
    <tr>
      <td>51</td>
      <td>6.4</td>
      <td>3.2</td>
      <td>4.5</td>
      <td>1.5</td>
      <td>0</td>
    </tr>
    <tr>
      <td>52</td>
      <td>6.9</td>
      <td>3.1</td>
      <td>4.9</td>
      <td>1.5</td>
      <td>0</td>
    </tr>
    <tr>
      <td>53</td>
      <td>5.5</td>
      <td>2.3</td>
      <td>4.0</td>
      <td>1.3</td>
      <td>0</td>
    </tr>
    <tr>
      <td>54</td>
      <td>6.5</td>
      <td>2.8</td>
      <td>4.6</td>
      <td>1.5</td>
      <td>0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>145</td>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
      <td>1</td>
    </tr>
    <tr>
      <td>146</td>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
      <td>1</td>
    </tr>
    <tr>
      <td>147</td>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>148</td>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>1</td>
    </tr>
    <tr>
      <td>149</td>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 5 columns</p>
</div>



```python
X = new_iris_df.drop('species', axis=1)
y = new_iris_df.species

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```


```python
binary_heap = construction_arbre(X_train, y_train, max_depth=3, gain_info_min = 0.01, indice='gini')
# 2.52s à construire l'arbre car valeur continue donc on doit évaluer chaque point de coupure (limitation youss)
```


```python
y_pred = [prediction(X_test.loc[i], binary_heap) for i in X_test.index]
```


```python
for (i, item) in enumerate(y_pred):
    if item < 0.5: # Point de coupure déterminé arbitrairement 
        y_pred[i] = 0
```


```python
pred_table = {'y':y_test, 'y_pred':y_pred}
```


```python
pred_table = pd.DataFrame(pred_table)
display(pred_table.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y</th>
      <th>y_pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>133</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>103</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>120</td>
      <td>1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>95</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>94</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
# On évalue la condition: si oui, on retourne 1, sinon 0
pred_table['prediction'] = np.where(pred_table['y'] == pred_table['y_pred'], 1, 0)
```


```python
print(f'Taux de bonne classification en test avec notre arbre : {pred_table.prediction.sum()/len(pred_table)}')
```

    Taux de bonne classification en test avec notre algo : 0.8


## Arbre de décision de sklearn 
**Avec les mêmes hyper paramètres que notre arbre de décision**


```python
X = new_iris_df.drop('species', axis=1)
y = new_iris_df.species

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```


```python
clf = DecisionTreeClassifier(max_depth=3, min_impurity_split=0.01) # On utilise les mêmes hyper paramètres
```


```python
clf.fit(X_train,y_train)
```

    /Users/florentfettu/opt/anaconda3/lib/python3.7/site-packages/sklearn/tree/_classes.py:301: FutureWarning: The min_impurity_split parameter is deprecated. Its default value will change from 1e-7 to 0 in version 0.23, and it will be removed in 0.25. Use the min_impurity_decrease parameter instead.
      FutureWarning)





    DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                           max_depth=3, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=0.01,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort='deprecated',
                           random_state=None, splitter='best')




```python
y_pred = clf.predict(X_test)
```


```python
print(f'Taux de bonne classification en test de l\'arbre de sklearn : {accuracy_score(y_test, y_pred)}')
# Meilleure performance en test de notre algo vs sklearn sur ce jeu de données là + mêmes hypers paramètres
```

    Taux de bonne classification en test de l'arbre de sklearn : 0.7666666666666667



```python

```
