import nltk
import pandas as pd
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

import networkx as nx
from collections import Counter
from itertools import chain


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

        : type str: Only put undirect or direct
        : window int: 1, 2
        : direction int: bidirection, forward, backward

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

    # 숫자로 된 노드와 이름속성 추
    for i, node_name in enumerate(idx_to_vocab):
        G.add_node(i, name=node_name)
    for (n1, n2), coor in coor_dict.items():
        G.add_edge(n1, n2)
    return G


if __name__ == "__main__":

    sentences = []
    with open('./Data/data2.txt') as f:
        lines = [line.rstrip() for line in f if line.rstrip() != '']
        for line in lines:
            sentences.extend(line.split('. '))
    print("Total Lines {0}".format(len(sentences)))

    stop_words_nltk = nltk.corpus.stopwords.words('english')
    snowball = nltk.stem.snowball.SnowballStemmer('english')
    lemm = nltk.stem.WordNetLemmatizer()

    df = pd.DataFrame()
    df["text"] = sentences
    df['tokenized_text'] = df['text'].apply(lambda x: nltk.word_tokenize(x))
    df['normalized_text'] = df['tokenized_text'].apply(
        lambda x: normlizeTokens(x, stopwordLst=stop_words_nltk, lemmer= lemm))

    tkn_cnt_lst = sum([len(tkn_cnt) for tkn_cnt in df['tokenized_text'].to_list()])
    print("Tokenized word count : {0}".format(tkn_cnt_lst))
    nor_cnt_lst = sum([len(nor_cnt) for nor_cnt in df['normalized_text'].to_list()])
    print("Normailized word count : {0}".format(nor_cnt_lst))

    sentences = df['normalized_text'].to_list()
    idx_to_vocab, vocab_to_idx = scan_vocabulary(sentences)
    print("Remove duplicated word {0}".format(len(idx_to_vocab)))

    #Graph test
    graph = word_graph(sentences, type="direct", window=2, direction='backward')





