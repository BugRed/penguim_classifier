# Import das bibliotecas

import pandas as pd # Biblioteca pandas para análise e manipulação de dados
import matplotlib.pyplot as plt # Biblioteca matplotlib para customizar gráficos
import seaborn as sns # Biblioteca Seaborn para esboçar gráficos
import numpy as np # Biblioteca numpy para cálculos matemáticos e vetorização
import graphviz # Biblioteca ghaphviz

# Exportar modelos treinados em pkl
import joblib

from sklearn.model_selection import train_test_split # Dividir os dados do conjunto em proporção de treinamento e testes
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz # Usar o modelo de árvore de decisão para classificação e desenhar a árvore
from sklearn.metrics import classification_report, confusion_matrix # Avaliar o desempenho do modelo
import streamlit as st


# ============================================================
# TABELA
# ============================================================

df_penguins_size = pd.read_csv('db/penguins_size.csv', sep=',')
df_penguins_size = df_penguins_size.dropna()


# df_penguins_size.info()

st.title('Análise Exploratória dos Pinguins')

with st.expander("Visualização da tabela de dados"):
    st.dataframe(df_penguins_size)


# ============================================================
# COUNT PLOT
# ============================================================

with st.expander("Gráfico de Contagem das Espécies"):
    fig1, ax1 = plt.subplots()
    sns.countplot(
        x='species',
        data=df_penguins_size,
        hue='species',
        ax=ax1,
        edgecolor='black',
        linewidth=1.5
    )
    ax1.set_title("Distribuição das Espécies de Pinguins")
    st.pyplot(fig1)


# ============================================================
# HEATMAP
# ============================================================

with st.expander("Heatmap das Correlações Numéricas"):
    X = df_penguins_size.drop(['species', 'island', 'sex'], axis=1)
    y = df_penguins_size['species']

    fig2, ax2 = plt.subplots()
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm', ax=ax2)
    ax2.set_title("Correlação entre Variáveis Numéricas")
    st.pyplot(fig2)


# ============================================================
# TREINO E TESTE
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=3
)

# ============================================================
# HISTOGRAMAS
# ============================================================

st.title("Histogramas e Violinplots das Variáveis Numéricas")

numeric_cols = X.columns

with st.expander("Histogramas"):
    for feature in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(
            df_penguins_size[feature],
            kde=True,
            bins=30,
            color='green',
            edgecolor='black',
            linewidth=1.2,
            ax=ax
        )
        ax.set_title(f"Distribuição de {feature}")
        st.pyplot(fig)
        plt.close(fig)

# ============================================================
# VIOLINPLOT + SWARMPLOT
# ============================================================

with st.expander("Violinplots e Swarmplots por Espécie"):
    for feature in numeric_cols:
        fig, ax = plt.subplots(figsize=(7, 5))

        sns.violinplot(
            data=df_penguins_size,
            x='species',
            y=feature,
            palette='muted',
            hue='species',
            inner='quartile',
            ax=ax
        )

        sns.swarmplot(
            data=df_penguins_size,
            x='species',
            y=feature,
            color='k',
            alpha=0.5
        )

        ax.set_title(f"Distribuição de {feature} por Espécie")
        ax.set_xlabel("Classe Predita (0 = Adelie, 1 = Chinstrap, 2 = Gentoo)")
        ax.set_ylabel(feature)

        st.pyplot(fig)
        plt.close(fig)

# ============================================================
# GRÁFICO DE DISPERSÃO ENTRE VARIAVEIS
# ============================================================

with st.expander("Gráfico de dispersão body_mass_g vs flipper_length_mm"):
        fig, ax = plt.subplots(figsize=(7, 5))

        sns.scatterplot(
            x='body_mass_g',
            y='flipper_length_mm',
            data=df_penguins_size,
            hue='species',
        )

        st.pyplot(fig)
        plt.close(fig)


## ============================================================
# INICIAR ALGORITMO DECISION TREE
# ============================================================

with st.expander("Treinamento do Modelo Decision Tree"):
    st.subheader("Treinando o modelo Decision Tree")

    tree_decision_classifier = DecisionTreeClassifier(random_state=0, criterion='gini')
    tree_decision_classifier = tree_decision_classifier.fit(X_train, y_train)

    y_pred = tree_decision_classifier.predict(X_test)

    st.success("Modelo treinado com sucesso!")

# ============================================================
# MÉTRICAS DE AVALIAÇÃO
# ============================================================

with st.expander("Relatório de Classificação"):
    st.subheader("Desempenho do Modelo")

    # Converte o classification_report para DataFrame mais bonito
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    st.dataframe(report_df.style.background_gradient(cmap="Greens"))

# ============================================================
# MATRIZ DE CONFUSÃO
# ============================================================

with st.expander("Matriz de Confusão"):
    st.subheader("Matriz de Confusão")

    cf = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cf,
        annot=True,
        cmap="Blues",
        fmt="d",
        linewidths=0.5,
        linecolor="black"
    )
    ax.set_title("Matriz de Confusão")
    ax.set_xlabel("Classe Predita (0 = Adelie, 1 = Chinstrap, 2 = Gentoo)")
    ax.set_ylabel("Classe Real")
    st.pyplot(fig)

# ============================================================
# NOVAS AMOSTRAS PARA PREDIÇÃO
# ============================================================

test_samples = df_penguins_size.sample(10, random_state=42)
test_samples_X = test_samples[X.columns] 
# Predição
test_preds = tree_decision_classifier.predict(test_samples_X)
test_proba = tree_decision_classifier.predict_proba(test_samples_X)

with st.expander("Predição para Novas Amostras"):
    st.subheader("Predição para Novas Amostras de Teste")

    # Adiciona coluna com a classe prevista
    test_samples['Predicted_Species'] = test_preds
    st.write("Classes previstas para as amostras:")
    st.dataframe(test_samples)

    # DataFrame com probabilidades
    proba_df = pd.DataFrame(
        test_proba,
        columns=tree_decision_classifier.classes_
    )
    st.write("Probabilidades de cada classe:")
    st.dataframe(proba_df.style.background_gradient(cmap="Greens"))


# ============================================================
# VISUALIZAÇÃO DA ARVORE DE DECISÃO
# ============================================================

with st.expander("Visualização da Árvore de Decisão"):
    st.subheader("Árvore de Decisão Treinada")

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        tree_decision_classifier,
        feature_names=X.columns,
        class_names=tree_decision_classifier.classes_,
        filled=True,
        rounded=True,
        fontsize=12,
        ax=ax
    )
    st.pyplot(fig)

# ============================================================
# EXPORTAR O MODELO TREINADO
# ============================================================

# Salva o modelo Decision Tree
joblib.dump(tree_decision_classifier, 'penguim_classifier_tree_model.pkl')

# Área de download do modelo no Streamlit
with st.expander("Download do Modelo Treinado"):
    st.subheader("Baixe o modelo Decision Tree treinado")
    
    with open("penguim_classifier_tree_model.pkl", "rb") as file:
        st.download_button(
            label="Baixar Modelo Decision Tree",
            data=file,
            file_name="penguim_classifier_tree_model.pkl",
            mime="application/octet-stream"
        )