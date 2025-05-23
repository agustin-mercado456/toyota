import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt


def dividir_datos(df: pd.DataFrame, test_size: float = 0.4, random_state: int = 42):

    X = df.drop(columns=["price"])
    y = df["price"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return x_train, x_test, y_train, y_test

def regresion_modelo(x_train, y_train):
   
    
    x_train = sm.add_constant(x_train)
    modelo = sm.OLS(y_train, x_train).fit()

        # Log de parámetros y métricas
    

    return modelo




