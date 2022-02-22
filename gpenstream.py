import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

penguins = pd.read_csv("penguins_size.csv")

penguins.drop(penguins[penguins['body_mass_g'].isnull()].index,axis=0, inplace=True)
penguins['sex'] = penguins['sex'].fillna('MALE')
penguins.drop(penguins[penguins['sex']=='.'].index, inplace=True)
a = penguins.isnull().sum()

df = penguins.copy()

target ='species'

s_mapper = {'MALE':0, 'FEMALE':1}
def s_encode(val):
    return s_mapper[val]

df['sex'] = df['sex'].apply(s_encode)

island_mapper = {'Torgersen':0, 'Biscoe':1, 'Dream' :2}
def island_encode(valu):
    return island_mapper[valu]

df['island'] = df['island'].apply(island_encode)


target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}
def target_encode(val):
    return target_mapper[val]

df['species'] = df['species'].apply(target_encode)
print(df.columns)

X = df.drop('species', axis=1)
Y = df['species']

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the model
import pickle
pickle.dump(clf, open('penguins_clfn.pkl', 'wb'))