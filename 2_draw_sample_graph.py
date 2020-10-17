import nltk
import pandas as pd
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
import networkx as nx
from collections import Counter
from itertools import chain
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout


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
                # 0에서 1인데, 1,

            for token2 in tokens[left_lim_idx:right_lim_idx]:
                if token1 != token2:
                    if type == "direct":
                        key = (token1, token2)
                    else:
                        key = tuple(sorted([token1, token2]))
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

def save_graph(graph,file_name):
    """
        Graph를 Plotting하고 저장한다.
    """

    pos = graphviz_layout(graph, prog='sfdp')
    names = nx.get_node_attributes(graph, 'name')
    nx.draw(graph, pos, node_size=4)
    nx.draw_networkx_labels(graph, pos,labels=names, font_size=10)
    plt.savefig(file_name, bbox_inches='tight')
    plt.clf()

def print_node_edge(graph, graph_name):
    """
        Graph의 node와 edge를 보여준다
    """
    print("{0} number_of_nodes : {1}".format(graph_name, graph.number_of_nodes()))
    print("{0} number_of_edges : {1}".format(graph_name, graph.number_of_edges()))

if __name__ == "__main__":
    sentences = []
    with open('./Data/sample_graph1.txt') as f:
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
    nor_cnt_lst = sum([len(nor_cnt) for nor_cnt in df['normalized_text'].to_list()])

    sentences = df['normalized_text'].to_list()
    idx_to_vocab, vocab_to_idx = scan_vocabulary(sentences)

    #draw graph
    graph_db2 = word_graph(sentences, type="direct", window=2, direction='backward')
    graph_db3 = word_graph(sentences, type="direct", window=3, direction='backward')
    graph_df2 = word_graph(sentences, type="direct", window=2, direction='forward')
    graph_df3 = word_graph(sentences, type="direct", window=3, direction='forward')
    graph_dbi2 = word_graph(sentences, type="direct", window=2, direction='bidirection')
    graph_dbi3 = word_graph(sentences, type="direct", window=3, direction='bidirection')
    graph_ub2 = word_graph(sentences, type="undirect", window=2, direction='backward')
    graph_ub3 = word_graph(sentences, type="undirect", window=3, direction='backward')
    graph_ubi2 = word_graph(sentences, type="undirect", window=2, direction='bidirection')
    graph_ubi3 = word_graph(sentences, type="undirect", window=3, direction='bidirection')

    #Draw sample graph
    save_graph(graph_db2, "d1_graph_db2.png")
    save_graph(graph_db3, "d1_graph_db3.png")
    save_graph(graph_df2, "d1_graph_df2.png")
    save_graph(graph_df3, "d1_graph_df3.png")
    save_graph(graph_dbi2, "d1_graph_dbi2.png")
    save_graph(graph_dbi3, "d1_graph_dbi3.png")
    save_graph(graph_ub2, "d1_graph_ub2.png")
    save_graph(graph_ub3, "d1_graph_ub3.png")
    save_graph(graph_ubi2, "d1_graph_ubi2.png")
    save_graph(graph_ubi3, "d1_graph_ubi3.png")

    print_node_edge(graph_db2, "d2_graph_db2")
    print_node_edge(graph_db3, "d2_graph_db3")
    print_node_edge(graph_df2, "d2_graph_df2")
    print_node_edge(graph_df3, "d2_graph_df3")
    print_node_edge(graph_dbi2, "d2_graph_dbi2")
    print_node_edge(graph_dbi3, "d2_graph_dbi3")
    print_node_edge(graph_ub2, "d2_graph_ub2")
    print_node_edge(graph_ub3, "d2_graph_ub3")
    print_node_edge(graph_ubi2, "d2_graph_ubi2")
    print_node_edge(graph_ubi3, "d2_graph_ubi3")