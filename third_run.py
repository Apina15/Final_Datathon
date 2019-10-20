import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

df = pd.read_csv('No_na.csv')
X = df.drop(columns=['target'])
y = df['target'].values


clas = RandomForestClassifier(random_state=random.seed(42), n_estimators=1000, n_jobs=-1, max_features=50)


clas.fit(X, y)
dump(clas, 'rfc_model.joblib')

