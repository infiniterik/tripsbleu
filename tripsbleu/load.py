import networkx as nx
import math

# Load a trips parse to networkx
# keep everything as attributes
# Sample node
# "V45214": {
#   "id": "V45214",
#   "indicator": "F",
#   "type": "HIGH-VAL",
#   "word": "HIGH",
#   "roles": {
#     "FIGURE": "#V45225",
#     "SCALE": "DIMENSIONAL-SCALE",
#     "LEX": "HIGH",
#     "WNSENSE": "high%3:00:02::"
#   },
#   "start": 59,
#   "end": 64
# }

def add_node(G, node, clone=True):
    """
    Inputs: G - a DiGraph, node - a dictionary
    Adds a node to a graph. The node key may or may not already exist

    if clone=False, will modify G
    if node["id"] already exists in G, then this will only *override* existing values

    This creates a small possible issue - what do you do if you want to entirely replace a node and all its values?
        Think about this when it comes to performing mutations
    """
    if clone:
        G = G.copy()
    id = node["id"]
    G.add_node(id)

    edges = {r: v for r, v in node.get("roles", {}).items() if v.startswith("#")}
    for r, v in edges.items():
        v = v[1:]
        G.add_edge(id, v, role=r)
    values = {r: v for r, v in node.get("roles", {}).items() if not v.startswith("#")}
    for k, v in node:
        if k != "roles":
            G.nodes[id][k] = v
    G.nodes[id]["values"] = {}
    for k, v in values.items():
        G.nodes[id]["values"][k] = v
    return G

def remove_node(G, id, clone=True):
    if clone:
        G = G.copy()
    G.remove_node(id)
    return G

def tripsnx(js, alt=None, version="alternatives"):
    """Loads an nx.DiGraph from a dict
    if alt is an integer, loads an alternative parse
    version says which key to look for alternatives in
    """
    if alt is not None:
        parse = js[version][alt]
    else:
        parse = js["parse"]
    roots = []
    G = nx.DiGraph()
    for subtree in parse:
        roots.append(subtree.get("root", "")[:1])
        for x, v in subtree.items():
            if x == "root":
                continue
            add_node(G, v, clone=False)
    return G, [r for r in roots if r]

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
    edge_path = []
    get_node = lambda x : G.nodes[x].get(node_label, x)
    for i in range(len(path)-1):
        edge_path.append(get_node(path[i]))
        edge_path.append(G.edges[(path[i], path[i+1])].get(edge_label))
    edge_path.append(get_node(path[-1]))
    return tuple(edge_path)

def kronecker(x, y):
    return int(x == y)

def l1(x, y, range_=1):
    return 1 - abs(x - y)/range_

def set_intersection(x, y):
    return len(set(x).intersection(set(y)))/len(set(y))

def weighted_ngram_score(edge=kronecker, node=kronecker, edge_weight=1, node_weight=1):
    def _ng_comp(a, b):
        if len(a) != len(b):
            raise ValueError("ngrams %s and %s are of different lengths" % (a, b))
        if len(a) % 2 == 0:
            raise ValueError("Got ngram of length %d, expected odd length" % len(a))
        comp = [_ for _ in zip(a, b)]
        score = 0
        for i in range(len(comp)):
            if i % 2 == 0:
                score += node(*comp[i]) * node_weight
            else:
                score += edge(*comp[i]) * edge_weight
        edges = len(a)//2
        nodes = len(a) - edges
        return score / (nodes * node_weight + edges * edge_weight)
    return _ng_comp

def greedy(score=weighted_ngram_score(), align=False):
    def _greedy(reference, candidate):
        scored = []
        for i, x in enumerate(reference):
            for j, y in enumerate(candidate):
                scored.append((i, j, score(x, y)))
        taken_x, taken_y = set(), set()
        alignments = []
        res = 0
        while scored:
            i, j, s = max(scored, key=lambda x: x[2])
            alignments.append((reference[i], candidate[j], i, j, s))
            res += s
            taken_x.add(i)
            taken_y.add(j)
            scored = [(i, j, s) for i, j, s in scored if i not in taken_x and j not in taken_y]
        if align:
            return res/len(candidate), aligned
        return res/len(candidate)
    return _greedy

def graph_length(G):
    return len(G.nodes) + len(G.edges)

def sembleu(reference, candidate, n=3, pk=set_intersection, weights=None, edge_label="role"):
    """
    reference and candidate are two parses
    n is the max length of ngrams
    eq is a function to test whether two
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
