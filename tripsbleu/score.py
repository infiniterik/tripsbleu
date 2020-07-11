import networkx as nx
import math

from .compare import set_intersection, weighted_ngram_score

def ngrams(G, n, edge_label="role", node_label="id"):
    """In order to use nx.all_simple_paths, we will copy G, add a source and sink,
    and compute all paths of length n+2
    """
    G = G.copy()

    nodes = [d for d in G.nodes]
    G.add_node("__NGRAM_SOURCE")
    G.add_node("__NGRAM_SINK")
    for d in nodes:
        G.add_edge("__NGRAM_SOURCE", d)
        G.add_edge(d, "__NGRAM_SINK")

    ng = sorted([_return_values(G, p[1:-1], edge_label, node_label) for p in nx.all_simple_paths(G, "__NGRAM_SOURCE", "__NGRAM_SINK", cutoff=n+2)], key=lambda x: len(x))
    # 2*i + 1 because we have edges as well
    return {i+1 : [x for x in ng if len(x) == (2*i + 1)] for i in range(n)}

def _return_values(G, path, edge_label="role", node_label="id"):
    """
    Take a path from a graph and return the sequence of labels
    if edge_label doesn't exist, the edge is labelled with None
    if node_label doesn't exist, the node is labelled with the vertex name
    """
    edge_path = []
    get_node = lambda x : G.nodes[x].get(node_label, x)
    for i in range(len(path)-1):
        edge_path.append(get_node(path[i]))
        edge_path.append(G.edges[(path[i], path[i+1])].get(edge_label))
    edge_path.append(get_node(path[-1]))
    return tuple(edge_path)


def graph_length(G):
    """
    :param G: a networkx graph
    :returns: the total size of the graph
    """
    return len(G.nodes) + len(G.edges)

def sembleu(reference, candidate, n=3, pk=set_intersection, weights=None, edge_label="role", node_label="id"):
    """
    :param reference: the reference graph
    :param candidate: the candidate graph
    :param pk: the function to compute pk for sembleu
    :param weights: weights[k-1] defines the contribution of k-grams to the score. If weights == None, ngrams are weighted uniformly
    :param n: the maximum length of ngrams to consider
    :param edge_label: edge_label to use for *ngrams* function
    :param node_label: node_label to use for *ngrams* function
    """
    # Brevity Penalty = e ^ (min(|r|/|c|, 0))
    #   |G| = |G.nodes| + |G.edges|
    BP = math.e ** min(graph_length(reference)/graph_length(candidate), 0)
    if not weights:
        weights = [1/n] * n
    referenceng = ngrams(reference, n, edge_label=edge_label)
    candidateng = ngrams(candidate, n, edge_label=edge_label)
    # What do you do when pk returns 0?
    # According to NLTK, if pk for ngrams of any order returns 0, the entire score is 0.
    pks = [pk(referenceng[i+1], candidateng[i+1]) for i in range(n)]
    for i, v in enumerate(pks):
        if v == 0:
            raise ValueError("%d-grams have no overlap" % (i+1))
            return 0
    return BP * (math.e ** sum([weights[i] * math.log(pks[i]) for i in range(n)]))
