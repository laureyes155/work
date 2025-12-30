'''
# 1. Importar librerías
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
from pydataset import data # Opcional: para cargar datos de ejemplo

from funciones import clean_dataset, clean_dataset_esp

fileNameParam='../datos/Spei_CodiVSTipoDeCambio_2019_2025.csv'
housing = read_csv(fileNameParam, usecols=[3], engine='python')
dataset0 = housing.values

dataset = clean_dataset_esp(dataset0,0)
print(dataset)

fileNameParam='../datos/Spei_CodiVSTipoDeCambio_2019_2021.csv'
housing = read_csv(fileNameParam, usecols=[3], engine='python')
dataset1 = housing.values

dataset = clean_dataset_esp(dataset0,0)
print(dataset)

fileNameParam='../datos/Spei_CodiVSTipoDeCambio_2022_2025.csv'
housing = read_csv(fileNameParam, usecols=[3], engine='python')
dataset2 = housing.values

dataset = clean_dataset_esp(dataset0,0)
print(dataset)

datos_a_plotear = [dataset0, dataset1, dataset2]
    # Crear el box plot
fig, ax = plt.subplots()
ax.boxplot(datos_a_plotear)

    # Agregar etiquetas y título
ax.set_title('Box plot simple')
ax.set_xlabel('Datos')
ax.set_ylabel('Valores')

    # Mostrar el gráfico
plt.show()
'''
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import read_csv

from funciones import clean_dataset, clean_dataset_esp, clean_dataset_esp2

# Leer solo la columna deseada para ahorrar memoria
df1 = clean_dataset_esp2(pd.read_csv('../datos/Spei_CodiVSTipoDeCambio_2019_2025.csv', usecols=[1], na_values=[ 'N/E']))
print('-------')
print(df1.values)
print('-------')
df2 = clean_dataset_esp2( pd.read_csv('../datos/Spei_CodiVSTipoDeCambio_2019_2021.csv', usecols=[1], na_values=['N/E']))
print('-------')
print(df2.values)
print('-------')
df3 = clean_dataset_esp2( pd.read_csv('../datos/Spei_CodiVSTipoDeCambio_2022_2025.csv', usecols=[1], na_values=[ 'N/E']))
print(df3.values.flatten())
print(type(df1.values))

# Find the maximum length
max_len = len(df1)
print(max_len)
# Pad shorter lists
# Pad shorter lists
if len(df2) < max_len:
    for i in range(len(df2), max_len):
        df2.loc[i]=np.nan

# Pad shorter lists
if len(df3) < max_len:
    for i in range(len(df3), max_len):
        df3.loc[i]=np.nan
print(max_len)
print(len(df2))
print(len(df3))
df = pd.DataFrame({
    '2019-2025': df1.values.flatten(),
    '2019-2021': df2.values.flatten(),
    '2022-2025': df3.values.flatten(),
})
print(df)

# Create boxplots for all three groups
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, color='white')
plt.title('SPEI amount')
plt.ylabel('Values')
plt.show()
print(df.describe())
