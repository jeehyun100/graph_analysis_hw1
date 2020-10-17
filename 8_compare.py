import nltk
import pandas as pd
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
import networkx as nx
from collections import Counter
from itertools import chain
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def normlizeTokens(tokenLst, stopwordLst=None, lemmer=None, vocab=None):
    """
        nltk를 사용하여  stemmer, lemmer를 사용하여 단어를 표준화를 하고, Stopword를 제거한다.

    """
    workingIter = (w.lower() for w in tokenLst if w.isalpha())
    if lemmer is not None:
        workingIter = (lemmer.lemmatize(w) for w in workingIter)
    if stopwordLst is not None:
        workingIter = (w for w in workingIter if w not in stopwordLst)
    return list(workingIter)

def scan_vocabulary(word_lst, min_count=1):
    """
        전체 문단에서 단어의 갯수를 세고, 빈도수가 많은 단어부터 차례대로 배열한다.
    """

    word_counter = Counter(chain.from_iterable(word_lst))
    word_counter = {w: c for w, c in word_counter.items() if c >= min_count}
    # idx_to_vocab: count가 높을 수록 앞에 등장
    idx_to_vocab = [w for w, c in sorted(
        word_counter.items(), key=lambda x: -x[1])]
    vocab_to_idx = {w: idx for idx, w in enumerate(idx_to_vocab)}
    return idx_to_vocab, vocab_to_idx

def vocab_cooccurrence_all(sentences, vocab_to_idx, direction, window=1, min_cooccurrence=1, type = "direct"):
    """
        방항에 따라 윈도우 사이즈 만큼 단어 pair를 만든다.
    """
    cooccurrence_dict = {}
    for words in sentences:
        tokens = [vocab_to_idx[t] for t in words if t in vocab_to_idx]
        if direction == 'backward':
            tokens.reverse()
        for i, token1 in enumerate(tokens):
            if direction == 'bidirection':
                left_lim_idx = max(0, i - window)
                right_lim_idx = min(len(tokens), i + window)
            elif direction == 'forward':
                left_lim_idx = max(0, i)
                right_lim_idx = min(len(tokens), i + window)
            elif direction == 'backward':
                left_lim_idx = max(0, i)
                right_lim_idx = min(len(tokens), i + window)

            for token2 in tokens[left_lim_idx:right_lim_idx]:
                if token1 != token2:
                    if type == "direct":
                        key = (token1, token2)
                    else:
                        key = tuple(sorted([token1, token2]))
                    # key = (token1, token2)
                    if key in cooccurrence_dict:
                        cooccurrence_dict[key] += 1
                    else:
                        cooccurrence_dict[key] = 1
    return {k: v for k, v in cooccurrence_dict.items() if v >= min_cooccurrence}

def word_graph(sentences, type="undirect", window=2, direction='bidirection',  min_count=1, min_cooccurrence=1):
    """
        Direct, Undirect Graph를 만든다

        :type str: Only put undirect or direct
        :window int: 1, 2
        :direction int: bidirection, forward, backward

    """
    idx_to_vocab, vocab_to_idx = scan_vocabulary(
        sentences, min_count=min_count)
    coor_dict = vocab_cooccurrence_all(
        sentences, vocab_to_idx, direction = direction, window=window, min_cooccurrence=min_cooccurrence)
    if type == "undirect":
        G = nx.Graph()
    elif type == "direct":
        G = nx.DiGraph()
    else:
        raise Exception("Graph type error")

    for i, node_name in enumerate(idx_to_vocab):
        G.add_node(i, name=node_name)
    for (n1, n2), coor in coor_dict.items():
        G.add_edge(n1, n2)
    return G

def open_txt():
    """
        문서 1 , 문서 2를 읽고 줄단위로 끊어서 리스트에 저장한다.
    """
    sentences1 = []
    sentences2 = []
    with open('./Data/data1.txt') as f:
        lines1 = [line.rstrip() for line in f if line.rstrip() != '']
        for line in lines1:
            sentences1.extend(line.split('. '))
    print("Total Lines {0}".format(len(sentences1)))

    with open('./Data/data2.txt') as f:
        lines2 = [line.rstrip() for line in f if line.rstrip() != '']
        for line in lines2:
            sentences2.extend(line.split('. '))
    print("Total Lines {0}".format(len(sentences2)))
    return sentences1, sentences2

def preprocessing_sentences(sentences):
    """
        기본 전처리를 하고 TF-IDF에 맞게 단어단위 list가 아니고 문장단위 list로 만든다.
    """
    df = pd.DataFrame()
    df["text"] = sentences
    df['tokenized_text'] = df['text'].apply(lambda x: nltk.word_tokenize(x))
    df['normalized_text'] = df['tokenized_text'].apply(
        lambda x: normlizeTokens(x, stopwordLst=stop_words_nltk, lemmer= lemm))

    tkn_cnt_lst = sum([len(tkn_cnt) for tkn_cnt in df['tokenized_text'].to_list()])
    print("Tokenized word count : {0}".format(tkn_cnt_lst))
    nor_cnt_lst = sum([len(nor_cnt) for nor_cnt in df['normalized_text'].to_list()])
    print("Tokenized word count : {0}".format(nor_cnt_lst))

    sentences = df['normalized_text'].to_list()
    idx_to_vocab, vocab_to_idx = scan_vocabulary(sentences)
    sentences = [' '.join(item) for item in sentences]

    return sentences, idx_to_vocab, vocab_to_idx


if __name__ == "__main__":
    sentences = []
    stop_words_nltk = nltk.corpus.stopwords.words('english')
    snowball = nltk.stem.snowball.SnowballStemmer('english')
    lemm = nltk.stem.WordNetLemmatizer()
    sentences1, sentences2 = open_txt()

    sentences1, idx_to_vocab1, vocab_to_idx1 = preprocessing_sentences(sentences1)
    sentences2, idx_to_vocab2, vocab_to_idx2 = preprocessing_sentences(sentences2)

    print("####Document1#####")
    tfidf = TfidfVectorizer(sublinear_tf = True)
    tfidf_matrix = tfidf.fit_transform(sentences1)


    scores = zip(tfidf.get_feature_names(),
                 np.asarray(tfidf_matrix.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    for item in sorted_scores[:6]:
        print ("{0:50} Score: {1}".format(item[0], item[1]))
    print("####Document2#####")
    tfidf = TfidfVectorizer(sublinear_tf = True)
    tfidf_matrix = tfidf.fit_transform(sentences2)

    scores = zip(tfidf.get_feature_names(),
                 np.asarray(tfidf_matrix.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    for item in sorted_scores[:6]:
        print ("{0:50} Score: {1}".format(item[0], item[1]))

