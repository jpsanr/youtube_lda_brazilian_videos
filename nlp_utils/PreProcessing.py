
from .Logger import Logger
from .Stacker import Stacker

from pydoc import doc
import spacy
import gensim
from typing import Union
from gensim import corpora
import re

class PreProcessing():

    def __init__(self, language:str="pt-br", stop_words_list:list=[]):
        """Construtor da classe PreProcessing. 

        Args:
            language (str, optional): idioma que será realizado os pré processamentos. Por padrão será setado "pt-br".
            stop_words_list (list, optional): Lista de palavras que serão tratadas como stop words. Por padrão recebe uma lista vazia.
        """
        if language == "en-us":
            spacy_model = "en_core_web_lg"
        else: 
            spacy_model = "pt_core_news_lg"

        self.logger = Logger()
        self.nlp = spacy.load(spacy_model) #Carregando Spacy
        self.stop_words = self.carregar_stop_words(stop_words_list) #Carregando stop words


    def carregar_stop_words(self, stop_words_list:list=[str], verbose=True):
        """Função que adiciona novas stop words ao obj self.nlp

        Args:
            stop_words_list (list, optional): Lista de palavras que serão tratadas como stop words. Por padrão recebe uma lista vazia.
        """
        if len(stop_words_list)>0:
            if verbose:
                tamanho_lista_original = len(list(self.nlp.Defaults.stop_words))
                self.logger.log("Adicionanado novas stopwords.")
                self.logger.log(f"   Serão adcionada(s) {len(stop_words_list)} palavra(s)")
                self.logger.log(f"   Tamanho da lista de stopwords: {tamanho_lista_original}")
            self.nlp.Defaults.stop_words |= set(stop_words_list)

            if verbose: 
                tamanho_lista_expandida = len(list(self.nlp.Defaults.stop_words))
                self.logger.log(f"   Tamanho da lista de stopwords: {tamanho_lista_expandida}")

    def tokenizar(self, lista_documentos:list, to_lower:bool=False, acentuacao:bool=True, verbose:bool=True) -> list:
        """Transforma o generator da função "sent_to_words" em uma lista

        Args:
            lista_documentos (list): Lista de string de textos
            to_lower (bool, optional): Se True mantém letras maiusculas. Por padrão False.
            acentuacao (bool, optional): Se False remove a acentuação das palavras. Ex: João -> Joao. Por padrão True
        Returns:
            list: Lista de tokens 
            verbose (bool, optional): Se True imprime logs. Por padrão True.
        """
        deacc = (not acentuacao)
        # lista_doc_tokenizado = list(list(gensim.utils.tokenize(documento, to_lower=to_lower, deacc=deacc)) for documento in lista_documentos)
        lista_doc_tokenizado = []
        for documento in lista_documentos:
            if documento is None:
                lista_doc_tokenizado.append(None)
                continue
            try:
                lista_doc_tokenizado.append(list(gensim.utils.tokenize(documento, to_lower=to_lower, deacc=deacc)))
            except Exception as e:
                if verbose:
                    self.logger.log(f"Tokenização:\n    ->Erro no documento: {documento}\n    ->Erro:\n{e}")
                lista_doc_tokenizado.append(None)
        return lista_doc_tokenizado

    def segmentar_sentencas(self, texto:str) ->list:
        """Dado um texto de entrada (com pontuações) é retornado uma lista de frases.

        Args:
            texto (str): texto a ser segmentado

        Returns:
            list: Lista de sentenças
        """

        doc = self.nlp(texto)   
        
        lista_sentencas = [sent for sent in doc.sents]

        return lista_sentencas

    def remover_urls(self, texto:str)->str:
        """Dado um texto de entrada, é removida todas as urls deste texto

        Args:
            texto (str): texto com urls

        Returns:
            str: texto sem urls
        """
        if texto is None: 
            return None

        url_pattern = r'((http|www)[(.|://)(\w]+)|(\w+\.com)'
        return re.sub(pattern=url_pattern, repl=' ', string=texto)

    def remover_urls_lista(self, lista_documentos:list[str])->list[str]:
        """A partir do método self.remover_urls(texto), remove as urls presentes em uma lista de documentos

        Args:
            lista_documentos (list[str]): Lista de documentos brutos. ATENCÃO: Esta lista NÃO deve ser tokenizada.

        Returns:
            list[str]: Lista de documentos sem urls.
        """
        
        lista_documentos = [self.remover_urls(documento) for documento in lista_documentos]

        return lista_documentos

    def remover_stop_words(self, lista_documentos_tokenizado:list)->list:
        """Remove os tokens considerados stop words de uma lista de tokens

        Args:
            lista_documentos_tokenizado (list): Lista de strings com os tokens de palavras. Para tokenizar um documento veja a função "tokenizar"

        Returns:
            list: Lista de tokens sem as stop words
        """

        lista_doc = []
        for sent in lista_documentos_tokenizado:
            
            if sent is None:
                lista_doc.append(None)
                continue

            lista_token = []
            for word in sent:
                lexeme = self.nlp.vocab[word]
                if lexeme.is_stop == False:
                    lista_token.append(word)
            lista_doc.append(lista_token)
        return lista_doc

    def filtrar_pos_tag(self, 
                        texto_ou_lista_token:Union[str, list], 
                        allowed_postags:list=["NOUN", "PROPN", "VERB", "ADJ"],
                        num_workers:int=1,
                        batch_size:int=10
                        )->Union[str, list]:
        """A partir de uma lista de tokens ou texto, filtra os tokens que não possuem as classes gramaticais passado no allowed_postags

        Args:
            texto (str | list): texto a ser processado ou lista de documentos
            allowed_postags (list, optional):Lista de pós tags. Por padrão ["NOUN", "PROPN", "VERB", "ADJ"].

        Returns:
            str | list: texto processado | lista de tokens
        """
        st = Stacker()

        if isinstance(texto_ou_lista_token, list):

            lista_token_entrada = st.remove_nones(texto_ou_lista_token)
            lista_token_saida = []
            for doc in self.nlp.pipe(lista_token_entrada, n_process=num_workers, batch_size=batch_size):
                saida = self.__filtrar_texto_pos_tag(doc, allowed_postags)
                
                lista_token_saida.append(saida)

            return st.add_nones(lista_token_saida)
        elif isinstance(texto_ou_lista_token, str):
            texto = texto_ou_lista_token
            return self._filtrar_texto_pos_tag(self.nlp(texto), allowed_postags)
        else:
            raise Exception("Erro no tipo de dado")



    def __filtrar_texto_pos_tag(self, doc_texto:spacy.tokens.doc.Doc, allowed_postags:list=["NOUN", "PROPN", "VERB", "ADJ"])->str:
        """A partir de um TEXTO, filtra os tokens que não possuem as classes gramaticais passado no allowed_postags

        Args:
            texto (str): texto a ser processado
            allowed_postags (list, optional): Lista de pós tags. Por padrão ["NOUN", "PROPN", "VERB", "ADJ"].

        Returns:
            str: texto processado
        """ 
        
        set_tokens = (str(token) if token.pos_ in allowed_postags else "" for token in doc_texto)
        
        texto_saida = " ".join(set_tokens)
        texto_saida = self.__tratar_espacos(texto_saida)
        return texto_saida

    def __tratar_espacos(self, texto:str):
        texto = re.sub(" +", " ",texto) #Removendo multiplos espaços
        if len(texto) >0:
            texto = texto[1:] if texto[0] == " " else texto #Remove primeiro espaço da str 
        if len(texto) >0:
            texto = texto[:-1] if texto[-1] == " " else texto #Remove ultimo espaço da str
        return texto

    def lematizar_documentos(self, lista_documentos_tokenizado:list, num_workers:int=1, batch_size:int=10, verbose:bool=True)->list:
        """A partir de uma lista de tokens retorna uma lista de tokens a partir dos lemmas

        Args:
            lista_documentos_tokenizado (list): Lista de strings com os tokens de palavras. Para tokenizar um documento veja a função "tokenizar"
            verbose (bool, optional): Se True imprime log. Por padrão True.
        Returns:
            list: Lista de tokens lematizados
        """

        lista_texto = [" ".join(item) if item is not None else None for item in lista_documentos_tokenizado]
        st = Stacker()
        lista_texto = st.remove_nones(lista_texto)

        documentos_out = []
        for doc in self.nlp.pipe(lista_texto, n_process=num_workers, batch_size=batch_size):
            saida = self.lematizar_texto(doc)
            documentos_out.append(saida)

        return st.add_nones(documentos_out)

    def lematizar_texto(self, doc:spacy.tokens.doc.Doc)->list:

        return [str(token.lemma_) for token in doc]

    def montar_n_grams(self, lista_documentos_tokenizado:list, min_count:int=5, threshold:float=10, grau:int=2)->tuple[list,gensim.models.phrases.Phrases]:
        """Facade que encapsula os métodos "treinar_modelo_ngram" e "encontrar_ngram_documento" permitindo o treinamento e classificação dos documentos de uma única vez.
        Args:
            lista_documentos_tokenizado (list): Lista de strings com os tokens de palavras.
            min_count (int, optional): Ignora Ngrams gerados com frenquência menor que este valor. Por padrão 5.
            threshold (float, optional): Confiança do ngrams gerados. Por padrão 10.
            grau (int, optional): Determina quantos gramas serão gerados. Ex: grau=2 -> bigrama, grau=3 -> trigrama. Por padrão 2.

        Raises:
            Exception: O valor de atributo grau deve ser >= 2.

        Returns:
            list:lista_documentos_tokenizado (list): Lista de strings com os tokens de palavras.
            gensim.models.phrases.Phrases: Modelo treinado de ngrams.
        """

        if grau <2:
            raise Exception("Grau deve ser >= 2")
        
        modelo_ngram = self.treinar_modelo_ngram(lista_documentos_tokenizado, min_count=min_count, threshold=threshold, grau=grau)
        lista_documentos_ngram = []
        for documento in lista_documentos_tokenizado:
            if documento is None:
                   lista_documentos_ngram.append(None)
                   continue
            lista_documentos_ngram.append(self.encontrar_ngram_documento(documento_tokenizado=documento,modelo=modelo_ngram))
        return lista_documentos_ngram, modelo_ngram

    def encontrar_ngram_documento(self, documento_tokenizado:list[str], modelo:gensim.models.phrases.Phrases)->list[str]:
        """A partir de um documento tokeninzado e um modelo de n_grams, retorna uma lista de tokens com os tokens ngrams, caso existam.

        Args:
            documento_tokenizado (list[str]): Lista de tokens (str).
            modelo (gensim.models.phrases.Phrases): Modelo de Ngrams. Ver o método "treinar_modelo_ngram".

        Returns:
            list[str]: Documento tokenizado.
        """
        if documento_tokenizado is None:
            return None
        return modelo[documento_tokenizado]

    def treinar_modelo_ngram(self, lista_documentos_tokenizado:list, min_count:int=5, threshold:float=10, grau:int=2)->gensim.models.phrases.Phrases:
        """A partir de uma lista de tokens, treina um modelo para encontrar os ocorrências mais comuns de tokens Ngrams. O número de tokens adjancentes que serão encontrados é determinado pelo atributo "grau".

        Args:
            lista_documentos_tokenizado (list): Lista de strings com os tokens de palavras.
            min_count (int, optional): Ignora Ngrams gerados com frenquência menor que este valor. Por padrão 5.
            threshold (float, optional): Confiança do ngrams gerados. Por padrão 10.
            grau (int, optional): Determina quantos gramas serão gerados. Ex: grau=2 -> bigrama, grau=3 -> trigrama. Por padrão 2.

        Raises:
            Exception: O valor de atributo grau deve ser >= 2.

        Returns:
            gensim.models.phrases.Phrases: Modelo treinado de ngrams.
        """

        if grau <2:
            raise Exception("Grau deve ser >= 2")

        lista_documentos_tokenizado = [documento for documento in lista_documentos_tokenizado if documento is not None]

        grau = grau-1
        lista_ngram = []
        for i in range(grau):
            if i == 0:
                #gerando bigram
                bigram = gensim.models.Phrases(lista_documentos_tokenizado, min_count=min_count, threshold=threshold) #Treinando Modelo
                lista_ngram.append(bigram)
                continue
            valor_gram = len(lista_ngram)-1
            ngram = gensim.models.Phrases(lista_ngram[valor_gram][lista_documentos_tokenizado], min_count=min_count, threshold=threshold)   #Treinando modelo
            lista_ngram.append(ngram)

        return lista_ngram[-1] #Retorna o modelo de ngram criado

    def get_frequencia_ngrams(self, lista_documentos_tokenizado:list)->dict:
        """Retorna um dicionário com todos os ngrams encontrados a partir de um corpus (lista de documentos tokenizado).
        Os valores deste dict correspondem a frequência deste token no corpus

        Args:
            lista_documentos_tokenizado (list): lista de documentos tokenizado

        Returns:
            dict: Tokens ngrams e sua frequência
        """
        dict_grams = {}

        lista_documentos_tokenizado = [documento for documento in lista_documentos_tokenizado if documento is not None]
        
        for doc in lista_documentos_tokenizado:
            if doc is None:
                continue
            for token in doc:
                if token.find("_") != -1:
                    if token in dict_grams:
                        dict_grams[token] = dict_grams[token] + 1
                    else:
                        dict_grams[token] = 1
                        
                    # lista_grams.append(token)
        return dict_grams

    def montar_id2word(self, lista_documentos_tokenizado:list)->gensim.corpora.dictionary.Dictionary:
        """Mapeia cada token de um corpus (lista de documentos tokenizado) e associa um id ao token.

        Args:
            lista_documentos_tokenizado (list): lista de documentos tokenizado

        Returns:
            gensim.corpora.dictionary.Dictionary: dicionário de ids com tokens
        """
        lista_documentos_tokenizado = [documento for documento in lista_documentos_tokenizado if documento is not None]
        id2word = corpora.Dictionary(lista_documentos_tokenizado)
        
        return id2word

    def montar_bow(self, lista_documentos_tokenizado:list,
                            id2word:gensim.corpora.dictionary.Dictionary)->list:
        """A partir de um corpus (lista de documentos tokenizado) e de um id2word retorna uma lista de BoW.

        Args:
            lista_documentos_tokenizado (list): lista de documentos tokenizado
            id2word (gensim.corpora.dictionary.Dictionary): dicionário de ids com tokens

        Returns:
            List: BoW
        """

        corpus = []
        for documento in lista_documentos_tokenizado:
            if documento is None:
                corpus.append(None)
                continue
            corpus.append(id2word.doc2bow(documento))
        return corpus
    
    def filtrar_extremos(self, id2word:gensim.corpora.dictionary.Dictionary, limite_inferior:int=5, limite_superior:float=0.5, n_tokens:int=100000)->gensim.corpora.dictionary.Dictionary:
        """Processa um id2word com o objetivo de filtrar tokens muito frequentes e não frequentes. Isto melhora a perfomance do algoritmo (menos consumo de memória) além de melhorar os resultados.
           Atenção: Por padrão é setado um numéro máximo de 100 mil tokens únicos na base. Por recomendação da biblioteca do Gensim este valor não deve ser ultrapassado.

        Args:
            id2word (gensim.corpora.dictionary.Dictionary): id2word
            limite_inferior (int, optional): Frequência mínima que um token deve aparecer para ser considerado válido. Por padrão 5.
            limite_superior (float, optional): Mantêm os tokens presentes em não mais que % do limite_superior. Por padrão 0.5
            n_tokens (int, optional): Número máxmo de tokens. Por padrão 100000.

        Returns:
            gensim.corpora.dictionary.Dictionary: id2word processado. ATENÇÃO: os ids originais foram alterados. Desta forma o BoW deve ser gerado novamente.
        """
        id2word.filter_extremes(no_below=limite_inferior, no_above=limite_superior,keep_n=n_tokens)
        return id2word

    def filtrar_token_tamanho(self, documento_tokenizado:list[str],tamanho_token:int=2)->list[str]:
        """Dado um documento tokenizado list[str], remove os tokens que são menores que o parâmetro "tamanho_token".

        Args:
            documento_tokenizado (list[str]): Documento tokenizado.
            tamanho_token (int, optional): Tamanho minímo de um token. Por padrão 2.

        Returns:
            list[str]: Documento tokenizado.
        """

        
        if documento_tokenizado is None:
            return None  
        doc = [] 
        for token in documento_tokenizado:
            if len(token) > tamanho_token:
                doc.append(token)
        return doc


    def pipeline_pre_processamento(
        self,
        lista_documentos:list[str],
        remover_url:bool=True, 
        remover_stop_word:bool=True, 
        filtar_pos_tag:bool=True, 
        lematizar:bool=True, 
        gerar_gramas:bool=True,
        filtrar_tamanho_token:bool=True,
        stop_word_list:list[str]=None,
        allowed_postags:list[str]=["NOUN", "PROPN", "VERB", "ADJ"],
        token_to_lower:bool=False,
        token_acentuacao:bool=True,
        n_gram_grau:int=2,
        n_gram_min_count:int=5,
        n_gram_threshold:int=10,
        tamanho_token:int=2,
        modelo_ngram:gensim.models.phrases.Phrases=None,
        batch_size:int=10,
        verbose:bool=True,
        num_workers=1
        )->tuple[list, gensim.models.phrases.Phrases]:
        """Método que encapsula todo o fluxo básico de pré-processamento em uma única função.
            Remoção de url >> Filtro de Pós Tag >> Tokenizar >> Remoção de Stop Words >> Lematização >> Geração de NGrams

        Args:
            lista_documentos (list[str]): Lista com os textos que serão processados.
            remover_url (bool, optional): Se True remove qualquer url contida nos textos. Por padrão True.
            remover_stop_word (bool, optional): Se True remove stop words. Por padrão  True.
            filtar_pos_tag (bool, optional): Se True filtra Pós Tags. Se True deve ser passado uma lista de Pós Tags no atributo "allowed_postags". Por padrão  True.
            lematizar (bool, optional): Se True irá lematizar os textos de entrada. Por padrão  True.
            gerar_gramas (bool, optional): Se True irá gerar Ngrams. Se True deve ser passado também os atributos "n_gram_grau", "n_gram_min_count", "n_gram_threshold". Por padrão  True.
            filtrar_tamanho_token (bool, optional): Se True filtra tokens menores que "tamanho_token". Por padrão True.
            stop_word_list (list[str], optional): Lista de stop words para ser expandida. Por padrão None.
            allowed_postags (list[str], optional):Lista de Pos tags. Por padrão ["NOUN", "PROPN", "VERB", "ADJ"].
            token_to_lower (bool, optional): _description_. Por padrão False.
            token_acentuacao (bool, optional): _description_. Por padrão True.
            n_gram_grau (int, optional): _description_. Por padrão 2.
            n_gram_min_count (int, optional): _description_. Por padrão 5.
            n_gram_threshold (int, optional): _description_. Por padrão 100.
            tamanho_token (int, optional): Tamanho minímo de um token. Por padrão 2.
            modelo_ngram(gensim.models.phrases.Phrases, optional): Caso um modelo Ngram seja passado, serão gerados os ngram dos documentos a patir de modelo. Caso não seja passado, será gerado um novo modelo Por padrão None.
            verbose (bool, optional): Se True imprime logs. Por padrão True.

        Returns:
            list: Lista de docoumentos tokenizados.
            gensim.models.phrases.Phrases: Modelo treinado de ngrams.
        """
        
        if verbose: 
            self.logger.log("Iniciando pipeline de pré processamento:")
          
        if remover_url:
            if verbose:
                self.logger.log(f"Removendo Urls:")
          
            lista_documentos = [self.remover_urls(documento) for documento in lista_documentos]

        if filtar_pos_tag:
            if verbose:
                self.logger.log("Filtrando Pos Tags")
            lista_documentos = self.filtrar_pos_tag(lista_documentos, allowed_postags=allowed_postags, num_workers=num_workers, batch_size=batch_size)

        if verbose: self.logger.log("Tokenizando")
        lista_documentos_tokenizado = self.tokenizar(lista_documentos, to_lower=token_to_lower, acentuacao=token_acentuacao, verbose=verbose)

        if remover_stop_word:
            if verbose:
                self.logger.log("Removendo Stop Words")
            if stop_word_list is not None:
                self.logger.log("    -> Expandindo Stop Words")
                self.carregar_stop_words(stop_words_list=stop_word_list, verbose=verbose)
            lista_documentos_tokenizado = self.remover_stop_words(lista_documentos_tokenizado)
        
        if lematizar:
            if verbose:
                self.logger.log("Lematizando")
            lista_documentos_tokenizado = self.lematizar_documentos(lista_documentos_tokenizado, verbose=verbose, num_workers=num_workers, batch_size=batch_size)
        
        if gerar_gramas:
            if verbose:
                if modelo_ngram is not None:
                    self.logger.log(f"Ngrams: Gerando Ngrams\n    -> Grau: {n_gram_grau}\n    -> Min Count: {n_gram_min_count}\n    -> Threshold: {n_gram_threshold}")
                else:
                    self.logger.log(f"Ngrams: Extraindo os Ngrams a partir de um modelo pré treinado.")
            if modelo_ngram is None:
                lista_documentos_tokenizado, modelo_ngram = self.montar_n_grams(lista_documentos_tokenizado, min_count=n_gram_min_count, threshold=n_gram_threshold, grau=n_gram_grau)
            else: 
                lista_documentos_tokenizado = [self.encontrar_ngram_documento(documento_tokenizado=documento, modelo=modelo_ngram) for documento in lista_documentos_tokenizado] 
        else:
            modelo_ngram = None

        if filtrar_tamanho_token:
            if verbose:
                self.logger.log(f"Filtrando tokens menores que {tamanho_token}")
            lista_documentos_tokenizado = [doc_tokenizado for doc_tokenizado in lista_documentos_tokenizado]
        
        return lista_documentos_tokenizado, modelo_ngram

if __name__ == "__main__":

    texto = "Joao mora com Maria. Maria tem um cachorro! O João anda de bicicleta?"
    lista_stop_words = ["Maria"]

    pp = PreProcessing(stop_words_list=lista_stop_words)

    
    
    print(pp.segmentar_sentencas(texto))