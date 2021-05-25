# Esse código utiliza como DB uma tradução que foi feita de reviews no IMDB e treina os dados de acordo com o que foi escrito para definir filmes 'ruins'(0) ou 'bons'(1)

from functions import *
import pandas as pd

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# # IMPORTANDO DADOS E TRATANDO DADOS
# acessando dados da planilha que foi baixada
dados = pd.read_csv("imdb-reviews-pt-br.csv")
# definindo coluna id como index
dados = dados.set_index('id')
# dropando coluna text_en
dados = dados.drop('text_en', 1)
# renomeando colunas para português
a_renomear = {
    'text_pt': 'texto_portugues',
    'sentiment': 'classificacao'
}
dados = dados.rename(columns=a_renomear)
# alterando dados da coluna classificacao para 0 ou 1
a_trocar = {'neg': 0, 'pos': 1}
dados.classificacao = dados.classificacao.map(a_trocar)

# RETIRANDO PALAVRAS IRRELEVANTES DO DF
stop_words = nltk.corpus.stopwords.words("portuguese")

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

dados.insert(1, "tratamento_1", frase_processada)
# print(dados)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==
# print(treino_teste_texto(dados, "tratamento_1", "classificacao"))

# nuvem_palavras(dados, "tratamento_1", 1)

# print(freq_palavras(dados, "tratamento_1", 1, 5))

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=