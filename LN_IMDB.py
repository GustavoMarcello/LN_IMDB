# Esse código utiliza como DB uma tradução que foi feita de reviews no IMDB e treina os dados de acordo com o que foi escrito para definir filmes 'ruins'(0) ou 'bons'(1)

from re import X
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

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# # IMPORTANDO DADOS E TRATANDO DADOS
# acessando dados da planilha que foi baixada
dados = pd.read_csv("imdb-reviews-pt-br.csv")
# print(dados)

# renomeando colunas para português
a_renomear = {
    'text_en': 'texto_inglês',
    'text_pt': 'texto_portugues',
    'sentiment': 'avaliacao'
}
dados = dados.rename(columns=a_renomear)

# definindo coluna id como index
dados = dados.set_index('id')

# convertendo dados da coluna "sentiment" alterando "neg" para 0 e "pos" para 1
classificacao = dados["avaliacao"].replace(["neg", "pos"], [0, 1])
# print(classificacao)

# criando coluna "classificação" com conteúdo binário de classificacao
dados["classificacao"] = classificacao
# print(dados)
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# # GERANDO MATRIZ ESPARÇA COM OS DADOS E TREINANDO MODELO
def classificar_texto(dados, feature, classe):
    # transformando a frase para um vetor de representação com tamanho máximo de n=50
    vetorizar = CountVectorizer(lowercase=False, max_features=50)
    bag_of_words = vetorizar.fit_transform(dados[feature])

    SEED = 42
    np.random.seed(SEED)

    # definindo variáveis de treino
    treino_x, teste_x, treino_y, teste_y = train_test_split(
        bag_of_words, dados[classe], stratify=dados[classe])

    regressao_logistica = LogisticRegression()            # set LogisticRegression()
    # treinou com o .fit()
    regressao_logistica.fit(treino_x, treino_y)
    acuracia = regressao_logistica.score(
        teste_x, teste_y)  # acurácia com o .score()
    return acuracia
# chamando função
# print(classificar_texto(dados, "texto_portugues", "classificacao"))

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# # GERANDO A IMAGEM DO WORDCLOUD
# Função filtra todos os dados negativos, e retorna a figura do wordcloud
def nuvem_palavras_negativas(dados, feature):
    #filtrando apenas por avaliações negativas
    texto_negativo = dados.query("classificacao == 0")
    # criando vetor com todas as palavras
    todas_palavras = ' '.join([texto for texto in texto_negativo[feature]])

    # criando a wordcloud e imprimindo a imagem
    # os parâmetros são opcionais, verificar documentação
    nuvem_palavras = WordCloud(width=800, height=500,
                            max_font_size=110, collocations=False).generate(todas_palavras)
    plt.figure(figsize=(10, 7))  # figsize define o tamanho da imagem
    # bilinear = mais nitida a img
    plt.imshow(nuvem_palavras, interpolation='bilinear')
    plt.axis("off") # retira os valores dos eixos x e y
    plt.show()

# Função filtra todos os dados positivos, e retorna a figura do wordcloud
def nuvem_palavras_positivas(dados, feature):
    #filtrando apenas por avaliações negativas
    texto_positivo = dados.query("classificacao == 1")
    # criando vetor com todas as palavras
    todas_palavras = ' '.join([texto for texto in texto_positivo[feature]])

    # criando a wordcloud e imprimindo a imagem
    # os parâmetros são opcionais, verificar documentação
    nuvem_palavras = WordCloud(width=800, height=500,
                            max_font_size=120, collocations=False).generate(todas_palavras)
    plt.figure(figsize=(10, 7))  # figsize define o tamanho da imagem
    # bilinear = mais nitida a img
    plt.imshow(nuvem_palavras, interpolation='bilinear')
    plt.axis("off") # retira os valores dos eixos x e y
    plt.show()

# nuvem_palavras_positivas(dados, "texto_portugues")
# nuvem_palavras_negativas(dados, "texto_portugues")

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#DEFININDO A FREQUENCIA DE CADA PALAVRA
#retorna um dicionário com a frequencia de cada palavra negativa do array
def freq_palavras_negativas(dados, feature):
    #filtrando apenas por avaliações negativas
    texto_negativo = dados.query("classificacao == 0")

    # criando vetor com todas as palavras
    todas_palavras = ' '.join([texto for texto in texto_negativo[feature]])

    #criando uma lista com cada palavra em um indice, separando por espaço
    token_espaco = tokenize.WhitespaceTokenizer()
    token_frase  = token_espaco.tokenize(todas_palavras)
    frequencia = nltk.FreqDist(token_frase)
    
    #criando df
    df_frequencia = pd.DataFrame({"Palavra": list(frequencia.keys()), 
                                  "Frequencia": list(frequencia.values())})
    return df_frequencia


#retorna um dicionário com a frequencia de cada palavra positiva do array
def freq_palavras_positivas(dados, feature):
    #filtrando apenas por avaliações negativas
    texto_positivo = dados.query("classificacao == 1")

    # criando vetor com todas as palavras
    todas_palavras = ' '.join([texto for texto in texto_positivo[feature]])

    #criando uma lista com cada palavra em um indice, separando por espaço
    token_espaco = tokenize.WhitespaceTokenizer()
    token_frase  = token_espaco.tokenize(todas_palavras)
    frequencia = nltk.FreqDist(token_frase)

    #criando df
    df_frequencia = pd.DataFrame({"Palavra": list(frequencia.keys()), 
                                  "Frequencia": list(frequencia.values())})
    return df_frequencia

# print(freq_palavras_negativas(dados, "texto_portugues"))
# print(freq_palavras_positivas(dados, "texto_portugues"))

#retornando as palavras positivas/negativas mais frequentes
frequentes_positivas = (freq_palavras_positivas(dados, "texto_portugues"))
# print(frequentes_positivas.nlargest(columns="Frequencia", n=10))
frequentes_negativas = (freq_palavras_negativas(dados, "texto_portugues"))
# print(frequentes_negativas.nlargest(columns="Frequencia", n=10))
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# GERANDO UM GRÁFICO COM AS PALAVRAS MAIS FREQUENTES

#retornando variável com as n mais frequentes

#plotando o grafico com as palavras POSITIVAS
def pareto_positivas(dados, feature, qtde_de_palavras):
    #filtrando apenas por avaliações negativas
    texto_positivo = dados.query("classificacao == 1")
    # criando vetor com todas as palavras
    todas_palavras = ' '.join([texto for texto in texto_positivo[feature]])

    #criando uma lista com cada palavra em um indice, separando por espaço
    token_espaco = tokenize.WhitespaceTokenizer()
    token_frase  = token_espaco.tokenize(todas_palavras)
    frequencia = nltk.FreqDist(token_frase)

    #definindo quantidade de palavras a serem vistas
    frequencia = frequentes_positivas.nlargest(columns="Frequencia", n=qtde_de_palavras)

    #plotando gráfico
    plt.figure(figsize=(12,8))
    ax = sns.barplot(data=frequencia, x="Palavra", y="Frequencia", color='gray')
    ax.set(ylabel = "Frequencia")
    plt.show()


#plotando o grafico com as palavras NEGATIVAS
def pareto_negativas(dados, feature, qtde_de_palavras):
    #filtrando apenas por avaliações negativas
    texto_negativo = dados.query("classificacao == 0")

    # criando vetor com todas as palavras
    todas_palavras = ' '.join([texto for texto in texto_negativo[feature]])

    #criando uma lista com cada palavra em um indice, separando por espaço
    token_espaco = tokenize.WhitespaceTokenizer()
    token_frase  = token_espaco.tokenize(todas_palavras)
    frequencia = nltk.FreqDist(token_frase)

    #definindo quantidade de palavras a serem vistas
    frequencia = frequentes_negativas.nlargest(columns="Frequencia", n=qtde_de_palavras)

    #plotando gráfico
    plt.figure(figsize=(12,8))
    ax = sns.barplot(data=frequencia, x="Palavra", y="Frequencia", color='gray')
    ax.set(ylabel = "Frequencia")
    plt.show()

# print(pareto_positivas(dados, "texto_portugues", 5))
# print(pareto_negativas(dados, "texto_portugues", 5))

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# # RETIRANDO PALAVRAS IRRELEVANTES DO DF

stop_words = nltk.corpus.stopwords.words("portuguese")
print(stop_words)

#criando uma lista 'frase_processada' sem as stopwords
token_espaco = tokenize.WhitespaceTokenizer()
frase_processada = list()

for i in dados.texto_portugues:
    nova_frase = list()
    palavras_texto = token_espaco.tokenize(i)
    for palavra in palavras_texto:
        if palavra not in stop_words:
            nova_frase.append(palavra)
    frase_processada.append(' '.join(nova_frase))

dados["tratamento_1"] = frase_processada    

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=



