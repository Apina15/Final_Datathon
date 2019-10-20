import pandas as pd
from joblib import load

df = pd.read_csv('No_na_test.csv')
clas_ = load('rfc_model.joblib')
df1 = pd.DataFrame(clas_.predict(df))
df1.to_csv('Final_Results.csv', index=False)


