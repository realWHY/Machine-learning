import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
from pdb import set_trace as bp

df = pd.read_excel('titanic.xls')
#print(df.head())
df.drop(['body','name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
#print(df.head())

def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            print('text_digit_vals: ',text_digit_vals)
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            print('column_contents: ',column_contents)
            unique_elements = set(column_contents)
            print('unique_elements: ',unique_elements)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    print('unique: ',unique)
                    text_digit_vals[unique] = x
                    print('text_digit_vals in : ',text_digit_vals)               
                    x+=1
            print('unique: ',unique)
            df[column] = list(map(convert_to_int, df[column]))
        
    return df

df = handle_non_numerical_data(df)
print(df)