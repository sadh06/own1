import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

penguins = pd.read_csv('penguins_size.csv')
penguins.drop(penguins[penguins['body_mass_g'].isnull()].index,axis=0, inplace=True)
penguins['sex'] = penguins['sex'].fillna('MALE')
penguins.drop(penguins[penguins['sex']=='.'].index, inplace=True)
penguins = penguins.drop(columns=['species'])

df = penguins.copy()

s_mapper = {'MALE':0, 'FEMALE':1}
def s_encode(val):
    return s_mapper[val]

df['sex'] = df['sex'].apply(s_encode)

island_mapper = {'Torgersen':0, 'Biscoe':1, 'Dream' :2}
def island_encode(valu):
    return island_mapper[valu]

df['island'] = df['island'].apply(island_encode)

df = df[:1] # Selects only the first row (the user input data)
print(df.columns)

# Reads in saved classification model
load_clf = pickle.load(open('penguins_clfn.pkl', 'rb'))

prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
print("done")
