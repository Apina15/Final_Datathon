import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


df = pd.read_csv('equip_failures_training_set.csv')
df = df.replace('na', np.NaN)
replacer = SimpleImputer(missing_values=np.NaN,  strategy='mean')
replacer = replacer.fit(df)
df1 = replacer.transform(df)
ans = pd.DataFrame(df1, columns=list(df)).to_csv('No_na.csv', index=False)

df3 = pd.read_csv('equip_failures_test_set.csv')
df3 = df3.replace('na', np.NaN)
replacer1 = SimpleImputer(missing_values=np.NaN,  strategy='mean')
replacer1 = replacer1.fit(df3)
df2 = replacer1.transform(df3)
ans1 = pd.DataFrame(df2, columns=list(df3)).to_csv('No_na_test.csv', index=False)






