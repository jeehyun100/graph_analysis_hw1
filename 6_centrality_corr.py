import nltk
import pandas as pd
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
import networkx as nx
from collections import Counter
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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

def get_corr_heatmap(graph, fix, ax, place, title, partial):

    if place == 1:
        subplot_w = ax[0,0]
    elif place == 2:
        subplot_w = ax[0,1]
    elif place == 3:
        subplot_w = ax[1,0]
    elif place == 4:
        subplot_w = ax[1,1]
    if partial == True:
        sorted_in_degree_centrality_g = [n for (n, c) in
                                                 sorted(nx.in_degree_centrality(graph).items(),
                                                        reverse=True, key=lambda item: item[1])[:10]]

        sorted_left_eigen_centrality_g = [n for (n, c) in
                                                  sorted(nx.eigenvector_centrality(graph).items(),
                                                         reverse=True, key=lambda item: item[1])[:10]]

        sorted_left_katz_centrality_g = [n for (n, c) in
                                                 sorted(nx.katz_centrality(graph, alpha=0.03).items(),
                                                        reverse=True, key=lambda item: item[1])[:10]]
        in_pagerank_g = [n for (n, c) in
                                      sorted(nx.pagerank(graph).items(),
                                             reverse=True, key=lambda item: item[1])[:10]]

        in_closeness_g = [n for (n, c) in
                                      sorted(nx.closeness_centrality(graph).items(),
                                             reverse=True, key=lambda item: item[1])[:10]]

        betweenness_g = [n for (n, c) in
                                      sorted(nx.betweenness_centrality(graph).items(),
                                             reverse=True, key=lambda item: item[1])[:10]]
    else:
        sorted_in_degree_centrality_g = [n for (n, c) in
                                         sorted(nx.in_degree_centrality(graph).items(),
                                                reverse=True, key=lambda item: item[1])]

        sorted_left_eigen_centrality_g = [n for (n, c) in
                                          sorted(nx.eigenvector_centrality(graph).items(),
                                                 reverse=True, key=lambda item: item[1])]

        sorted_left_katz_centrality_g = [n for (n, c) in
                                         sorted(nx.katz_centrality(graph, alpha=0.03).items(),
                                                reverse=True, key=lambda item: item[1])]
        in_pagerank_g = [n for (n, c) in
                         sorted(nx.pagerank(graph).items(),
                                reverse=True, key=lambda item: item[1])]

        in_closeness_g = [n for (n, c) in
                          sorted(nx.closeness_centrality(graph).items(),
                                 reverse=True, key=lambda item: item[1])]

        betweenness_g = [n for (n, c) in
                         sorted(nx.betweenness_centrality(graph).items(),
                                reverse=True, key=lambda item: item[1])]


    data_centrality = pd.DataFrame()
    data_centrality = data_centrality.assign(degree = sorted_in_degree_centrality_g)
    data_centrality = data_centrality.assign(eigen=sorted_left_eigen_centrality_g)
    data_centrality = data_centrality.assign(katz=sorted_left_katz_centrality_g)
    data_centrality = data_centrality.assign(pagerank=in_pagerank_g)
    data_centrality = data_centrality.assign(closeness=in_closeness_g)
    data_centrality = data_centrality.assign(betwennness=betweenness_g)

    corr_df = data_centrality.corr()

    mask = np.zeros_like(corr_df, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # 히트맵을 그린다
    subplot_w.set_title(title, fontdict={'fontsize': 12, 'fontweight': 'bold'})
    h1 = sns.heatmap(corr_df,
                cmap='RdYlBu_r',
                annot=True,  # 실제 값을 표시한다
                #annot_kws = {"size": 5},
                mask=mask,  # 표시하지 않을 마스크 부분을 지정한다
                linewidths=.5,  # 경계면 실선으로 구분하기
                cbar=False,
                #cbar_kws={"shrink": .5},  # 컬러바 크기 절반으로 줄이기
                vmin=-1, vmax=1,  # 컬러바 범위 -1 ~ 1
                ax = subplot_w
                )
    h1.set_yticklabels(h1.get_yticklabels(), size=10)
    h1.set_xticklabels(h1.get_xticklabels(), size=10)

if __name__ == "__main__":

    sentences = []
    with open('./Data/data1.txt') as f:
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
    print("Tokenized word count : {0}".format(nor_cnt_lst))

    sentences = df['normalized_text'].to_list()
    idx_to_vocab, vocab_to_idx = scan_vocabulary(sentences)

    #back
    graph_db2 = word_graph(sentences, type="direct", window=2, direction='backward')
    graph_db3 = word_graph(sentences, type="direct", window=3, direction='backward')
    graph_df2 = word_graph(sentences, type="direct", window=2, direction='forward')
    graph_df3 = word_graph(sentences, type="direct", window=3, direction='forward')
    graph_dbi2 = word_graph(sentences, type="direct", window=2, direction='bidirection')
    graph_dbi3 = word_graph(sentences, type="direct", window=3, direction='bidirection')

    fig, ax = plt.subplots(2,2, figsize=(14, 14))
    get_corr_heatmap(graph_df2, fig, ax,1, "n+1 graph", True)
    get_corr_heatmap(graph_df3, fig, ax,2, "n+2 graph", True)
    get_corr_heatmap(graph_dbi2, fig, ax,3, "n-1&n+1 graph", True)
    get_corr_heatmap(graph_dbi3, fig, ax,4, "n-1&n+1 graph", True)

    plt.savefig("d1_corr_centrality_2.png",bbox_inches='tight')
    plt.clf()

    fig, ax = plt.subplots(2, 2, figsize=(14, 14))
    get_corr_heatmap(graph_df2, fig, ax, 1, "n+1 graph", False)
    get_corr_heatmap(graph_df3, fig, ax, 2, "n+2 graph", False)
    get_corr_heatmap(graph_dbi2, fig, ax, 3, "n-1&n+1 graph", False)
    get_corr_heatmap(graph_dbi3, fig, ax, 4, "n-1&n+1 graph", False)

    plt.savefig("d1_corr_centrality_1.png", bbox_inches='tight')
    plt.close()