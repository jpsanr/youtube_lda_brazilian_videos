# -*- coding: utf-8 -*-
from .Logger import Logger

from typing import Union

import pandas as pd

import matplotlib.pyplot as plt

import gensim
from gensim import corpora, models, similarities
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import CoherenceModel, LdaModel


class TopicModeling():

    def __init__(self, corpus:list=None, id2word:gensim.corpora.dictionary.Dictionary=None):
        """Classe responsável geração de modelos de tópicos, a partir do algoritmo LDA.

        Args:
            corpus (list, optional): Representação vetorial dos documentos. Ex: Bag of Words.
            id2word (gensim.corpora.dictionary.Dictionary, optional): dicionário de ids com tokens.
        """
        self.logger = Logger()
        self.corpus = corpus
        self.id2word = id2word


    def treinar_multiplos_modelos(
        self, 
        vetor_corpus:list,
        id2word:gensim.corpora.dictionary.Dictionary, 
        lista_documentos_tokenizado:list,
        limit:int=10,
        start:int=2,
        step:int=3, 
        random_state:int=1,
        verbose:int=True
        )->list:
        """Gera multiplos modelos de LDA, de forma incremental, começando com um k="limit", até um k="max", incrementando com parâmetro "step"

        Args:
            id2word (gensim.corpora.dictionary.Dictionary): dicionário de ids com tokens
            vetor_corpus (list): representação vetorial do corpus utilizado.
            lista_documentos_tokenizado (list): lista de documentos tokenizado. 
            limit (int, optional): Número máximo de tópicos. Por padrão 10.
            start (int, optional): Número mínimo de tópicos. Por padrão 2.
            step (int, optional): Taxa incremental de tópicos. Por padrão 3.
            verbose (bool, optional): Se True imprime logs. Por padrão True.

        Returns:
            list: Lista de dicionários contendo os modelos, coerencia e num de tópicos
        """
        if verbose:
            self.logger.log("Start: {}".format(start)) 
            self.logger.log("limit: {}".format(limit))
            self.logger.log("Step: {}".format(step))       
        self.start = start 
        self.limit = limit
        self.step = step
        
        #Removendo itens None
        vetor_corpus = [item for item in vetor_corpus if item is not None]
        lista_documentos_tokenizado = [documento for documento in lista_documentos_tokenizado if documento is not None]

        lista_dict_modelos = []
        for num_topics in range(start, limit, step):
            if verbose:
                self.logger.log(f"Gerando novo modelo: K={num_topics}")
           
            model = LdaModel(corpus=vetor_corpus, 
                     num_topics=num_topics,
                     id2word=id2word,
                     random_state=random_state

            )
            coherencemodel = CoherenceModel(model=model, texts=lista_documentos_tokenizado, dictionary=id2word, coherence='c_v')
            
            dict_modelo = {"modelo": model, "coerencia":coherencemodel.get_coherence(), "num_topico":num_topics} 
            lista_dict_modelos.append(dict_modelo)
        
        
        if verbose:
            #TODO Informar qual o melhor modelo, sua coerencia, e index da lista
            pass

        self.lista_dict_modelos = lista_dict_modelos      
        return lista_dict_modelos

    def treinar_modelo(
        self, 
        corpus:list,
        num_topics:int,
        id2word:gensim.corpora.dictionary.Dictionary,
        calcular_coerencia:bool=False,
        lista_documentos_tokenizado:list=None,
        random_state:int=1
        )->tuple[gensim.models.ldamodel.LdaModel,Union[None, float]]:
        """Treina um modelo LDA a partir de um corpus(lista de documentos tokenizados) e seu respectivo id2word.

        Args:
            corpus (list): Representação vetorial dos documentos. Ex: Bag of Words.
            id2word (gensim.corpora.dictionary.Dictionary): dicionário de ids com tokens.
            num_topics (int): Número de tópicos a ser gerado.
            calcular_coerencia (bool, optional): Se True, calcula a coerência do modelo. Por padrão False.
            lista_documentos_tokenizado (list, optional): Lista de strings com os tokens de palavras. Se calcular_coerencia for False, nada acontece. Por padrão None.

        Returns:
            tuple[gensim.models.ldamodel.LdaModel,list]: Retorna o modelo LDA e caso calcular_coerencia==True, retorna a sua coerência também.
        """

        model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word, random_state=random_state)
        
        valor_coerencia = None
        if calcular_coerencia:
            valor_coerencia = CoherenceModel(model=model, texts=lista_documentos_tokenizado, dictionary=id2word, coherence='c_v').get_coherence()

        return model, valor_coerencia

    def retornar_melhor_modelo(self, lista_dict_modelos:list=None)->dict:
        
        if lista_dict_modelos is None:
            lista_dict_modelos = self.lista_dict_modelos
            
        lista_ordenada = sorted(lista_dict_modelos, key=lambda d: d['coerencia'],reverse=True) 
        self.dict_melhor_modelo = lista_ordenada[0]
        return lista_ordenada[0]

    def retornar_top_key_words(self, modelo:gensim.models.ldamodel.LdaModel, num_palavras:int=30)->pd.DataFrame:
        dict_palavras_topicos = {}

        for index, topic in modelo.show_topics(num_topics=-1,num_words=num_palavras,formatted=False):
            dict_words = {}
            for i, palavra in enumerate(topic,start=1):
                dict_words["palavra_{}".format(i)] = palavra[0]
                dict_words["prob_{}".format(i)] = float(palavra[1])
                #print("Palavra: {} - Peso: {}".format(palavra[0],palavra[1]))
            dict_words["topico"] = index
            dict_palavras_topicos["topico_"+str(index)] = dict_words     
        df_palavras = pd.DataFrame.from_dict(dict_palavras_topicos, orient='index')

        return df_palavras, dict_palavras_topicos
    
    def retornar_top_key_words_agrupado(self, modelo, num_palavras=30):
        dict_palavras = {"topico_id":[], "palavra":[], "peso":[]}
        for topico in modelo.show_topics(num_topics=-1,num_words=num_palavras,formatted=False):
            topico_id = topico[0]
            for tupla_palavra in topico[1]:
                palavra = tupla_palavra[0]
                peso = tupla_palavra[1]
                dict_palavras["topico_id"].append(topico_id)
                dict_palavras["palavra"].append(palavra)
                dict_palavras["peso"].append(peso)
        return pd.DataFrame(dict_palavras)

    def plotar_coerencia(self):
        x = range(self.start, self.limit, self.step)
        plt.plot(x, self.coherence_values)
        plt.xlabel("Num de Tópicos")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc='best')
        plt.show()
        for m, cv in zip(x, self.coherence_values):
            print("Num de Tópicos =", m, " valor coerência: ", round(cv, 4))

    def __aux_processar_topico_prob(self, item, grau):
        try:
            if item[0] is None:
                return None, None
            if len(item) > grau:
                topico = item[grau][0]
                probabilidade = item[grau][1]
            else:
                # print(f"grau: {grau}\n    {item}")
                return None, None
        except Exception as e:
            print(e)
            print(item)
            return None, None
        return topico, probabilidade



    def processar_df_topicos_probabilidade(self, df):
        '''busca os 4 principais tópicos e salva em colunas'''
        
        df['topico_1'] = df.apply(lambda x: self.__aux_processar_topico_prob(x['lista_topicos'],grau=0)[0], axis=1)
        df['topico_2'] = df.apply(lambda x: self.__aux_processar_topico_prob(x['lista_topicos'],grau=1)[0], axis=1)
        df['topico_3'] = df.apply(lambda x: self.__aux_processar_topico_prob(x['lista_topicos'],grau=2)[0], axis=1)
        df['topico_4'] = df.apply(lambda x: self.__aux_processar_topico_prob(x['lista_topicos'],grau=3)[0], axis=1)
        
        df['prob_1'] = df.apply(lambda x: self.__aux_processar_topico_prob(x['lista_topicos'],grau=0)[1], axis=1)
        df['prob_2'] = df.apply(lambda x: self.__aux_processar_topico_prob(x['lista_topicos'],grau=1)[1], axis=1)
        df['prob_3'] = df.apply(lambda x: self.__aux_processar_topico_prob(x['lista_topicos'],grau=2)[1], axis=1)
        df['prob_4'] = df.apply(lambda x: self.__aux_processar_topico_prob(x['lista_topicos'],grau=3)[1], axis=1)

        df['topico_2'] = df['topico_2'].astype(dtype=pd.Int64Dtype())
        df['topico_3'] = df['topico_3'].astype(dtype=pd.Int64Dtype())
        df['topico_4'] = df['topico_4'].astype(dtype=pd.Int64Dtype())

        return df
    
    def processar_df_topicos_probabilidade_one_hot_encoding(self, df):
        lista_dict = []
        for row in df.iterrows():
            lista_tuplas = row[1]["lista_topicos"][1]
            chamado_id = row[1]["cod_id"]
            aux_dict = {}
            for t in lista_tuplas:
                
                aux_dict[f"topico_{t[0]}"] = t[1]
            aux_dict["cod_id"] = chamado_id
            lista_dict.append(aux_dict)
            

        df_one_hot = pd.DataFrame(lista_dict)
       
        
        return df_one_hot

    def classificar_novo_vetor(self, doc_vetorizado, modelo):

        if doc_vetorizado is None:
            return None,None

        topicos = modelo[doc_vetorizado]
        
        #topicos_ordenados = sorted(topicos[0], key=lambda x: x[1], reverse=True)
        topicos_ordenados = sorted(topicos, key=lambda x: x[1], reverse=True)
        # melhor_topico = topicos_ordenados[0]
        #print(topicos_ordenados)
        if type(topicos_ordenados) is tuple:
            # print("temos uma tupla!")
            print(topicos_ordenados)

        return topicos_ordenados

if __name__ == "__main__":
   tm = TopicModeling()
   print(tm.teste("oi"))