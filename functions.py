import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk import tokenize
from wordcloud import WordCloud
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


# # TREINANDO MODELO
def treino_teste_texto(dados, feature, classificacao):
    # transformando a frase para um vetor de representação com tamanho máximo de n=50
    vetorizar = CountVectorizer(lowercase=False, max_features=50)
    bag_of_words = vetorizar.fit_transform(dados[feature])

    SEED = 42
    np.random.seed(SEED)
    treino_x, teste_x, treino_y, teste_y = train_test_split(
        bag_of_words, dados[classificacao], stratify=dados[classificacao])
    regressao_logistica = LogisticRegression()
    regressao_logistica.fit(treino_x, treino_y)
    acuracia = regressao_logistica.score(
        teste_x, teste_y)
    return acuracia



# # GERANDO A IMAGEM DO WORDCLOUD
def nuvem_palavras(dados, feature, classe):
    # filtrando apenas por avaliações negativas
    texto = dados.query(f"classificacao == {classe}")
    # criando vetor com todas as palavras
    todas_palavras = ' '.join([texto for texto in texto[feature]])

    # criando a wordcloud e imprimindo a imagem
    # os parâmetros são opcionais, verificar documentação
    nuvem_palavras = WordCloud(width=800, height=500,
                            max_font_size=110, collocations=False).generate(todas_palavras)
    plt.figure(figsize=(10, 7))  # figsize define o tamanho da imagem
    # bilinear = mais nitida a img
    plt.imshow(nuvem_palavras, interpolation='bilinear')
    plt.axis("off")  # retira os valores dos eixos x e y
    plt.show()



# DEFININDO A FREQUENCIA DE CADA PALAVRA E PLOTANDO UM GRÁFICO
def freq_palavras(dados, feature, classe, qtde_de_registros):
    # filtrando apenas por avaliações negativas
    texto = dados.query(f"classificacao == {classe}")

    # criando vetor com todas as palavras
    todas_palavras = ' '.join([texto for texto in texto[feature]])

    # criando uma lista com cada palavra em um indice, separando por espaço
    token_espaco = tokenize.WhitespaceTokenizer()
    token_frase = token_espaco.tokenize(todas_palavras)
    registros = nltk.FreqDist(token_frase)

    # criando df
    df_frequencia = pd.DataFrame({"Palavra": list(registros.keys()),
                                "Frequencia": list(registros.values())})
    df_frequencia = df_frequencia.sort_values(
        by=['Frequencia'], ascending=False)
    print(df_frequencia.head(qtde_de_registros))

    #definindo quantidade de palavras a serem vistas
    registros = df_frequencia.nlargest(columns="Frequencia", n=qtde_de_registros)
    #plotando gráfico
    plt.figure(figsize=(12,8))
    ax = sns.barplot(data=registros, x="Palavra", y="Frequencia", color='gray')
    ax.set(ylabel = "Frequencia")
    plt.show()

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


# # PLOTANDO GRÁFICO ###PRECISA SER AJUSTADO NA VARIÁVEL df_frequencia
# def pareto(dados, feature, qtde_de_palavras, classe):
#     #filtrando apenas por avaliações negativas
#     texto = dados.query(f"classificacao == {classe}")
#     # criando vetor com todas as palavras
#     todas_palavras = ' '.join([texto for texto in texto[feature]])

#     #criando uma lista com cada palavra em um indice, separando por espaço
#     token_espaco = tokenize.WhitespaceTokenizer()
#     token_frase  = token_espaco.tokenize(todas_palavras)
#     frequencia = nltk.FreqDist(token_frase)

#     #definindo quantidade de palavras a serem vistas
#     frequencia = df_frequencia.nlargest(columns="Frequencia", n=qtde_de_palavras)

#     #plotando gráfico
#     plt.figure(figsize=(12,8))
#     ax = sns.barplot(data=frequencia, x="Palavra", y="Frequencia", color='gray')
#     ax.set(ylabel = "Frequencia")
#     plt.show()
