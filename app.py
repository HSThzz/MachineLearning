import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import streamlit as st

# Lê os dados
df = pd.read_csv("pizzas.csv")

# Treina o modelo
x = df[["diametro"]]
y = df[["preco"]]
modelo = LinearRegression()
modelo.fit(x, y)

# Interface do Streamlit
st.title("Prevendo o valor de uma pizza 🍕")
st.divider()

diametro = st.number_input("Digite o tamanho da pizza (cm):", min_value=1.0, step=1.0)

if diametro > 0:
    entrada = pd.DataFrame([[diametro]], columns=["diametro"])
    preco_previsto = modelo.predict(entrada)
    st.write(f"O valor estimado da pizza com {diametro} cm é: **R$ {preco_previsto[0][0]:.2f}**")

