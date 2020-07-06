from pprint import pprint
import math
import json
from munkres import Munkres

class GramPair:
    """
    A gram is a combination of a node and an incoming edge.
    the default incoming edge is "ROOT" and can be ignored
    """
    def __init__(self, node, edge="ROOT", graph_type="trips", id=None):
        self.node = node
        self.edge = edge
        self.graph_type = graph_type
        self.id = id

    def __repr__(self):
        return "%s->%s" % (self.edge, self.node["id"])

    def node_(self, node_type="type"):
        return GramElement(self.node, node_type, id=self.id)

    def edge_(self):
        return GramElement(self.edge, "edge")

    def _flatten(self, node_type="type"):
        return [GramElement(self.edge, "edge"), GramElement(self.node, node_type, id=self.id)]

    @staticmethod
    def flatten(lst, node_type="type", edge_type="edge"):
        return [sum([x._flatten(node_type) for x in y], []) for y in lst]

class GramElement:
    """
    Type asserting wrapper for labels in NGrams.
    Edges and Nodes are treated differently.  May even want a mechanism for comparing different
    types of objects
    """
    def __init__(self, value, type_, extract=None, id=None):
        """
        """
        self.value = value
        self.type = type_
        self.extract = extract
        self.id = id

    @property
    def label(self):
        if self.extract:
            return self.extract(self.value)
        if self.type == "edge":
            return self.value
        if self.type == "type":
            if type(self.value) is str: #non-id nodes
                return self.value
            return self.value["type"]
        if self.type == "word":
            return self.value["word"]
        if self.type == "lex":
            return self.value["roles"]["LEX"]
        return None

    def __repr__(self):
        return "GE({}.{}.{})".format(self.label, self.type, (self.id or ""))

    def __str__(self):
        return repr(self)

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, o):
        if isinstance(o, GramElement):
            return self.type == o.type and self.label == o.label and self.label and o.label
        return False

def _node_sim(a, b, func="wup"):
    """
    Compute the score between to trips "type" GramElements
    """
    if (a.label == "ROOT") or (b.label == "ROOT") or (a.type != "type") or (b.type != "type"):
        return float(a.label == b.label)
    #if (type(a.label) is str) or (type(b.label) is str): #non-id nodes - deal with this better. there are extractions to be made in flatten
    #    return a.label == b.label
    al = ont[a.label]
    bl = ont[b.label]

    if not al or not bl:
        return float(a.label == b.label)

    if func == "wup":
        return al.wup(bl)
    elif func == "cosine":
        return al.cosine(bl)
    elif func == "eq" or func is None:
        return float(al == bl)

def node_sim(func="wup"):
    """
    func types are "wup", "cosine", and "eq"
    """
    return lambda a, b: _node_sim(a, b, func=func)


def ngrams(graph, n=1, graph_type="trips", skip=None):
    """
    An ngram is defined as a path containing n nodes
    graph: a graph object for the corresponding type
    n: the number of nodes to include 
    """
    if graph_type == "trips":
        return ngrams_trips(graph, n=n, skip=skip)
    else:
        return []

def ngrams_trips(graph, n=1, skip=None):
    """
    Turn a trips graph into a list of ngrams of length n.
    input is json_format["parse"]
    TODO: Transform parses into a better graph representation so that non-id nodes can be included or excluded as needed
    """
    if type(graph) is list:
        # flatten the graph and ignore root keys
        new_graph = {}
        for g in graph:
            for x, y in g.items():
                new_graph[x] = y
        return ngrams_trips(new_graph, n=n, skip=skip)
    else:
        # we are ignoring the root key and starting from each node.
        res = [ngrams_from_node_trips(graph, x, role="ROOT", n=n, skip=skip) for x in graph if x != "root"]
        return sum([r for r in res if r], []) # get rid of empties

def ngrams_from_node_trips(graph, root, role="ROOT", n=1, skip=None):
    """
    recursively find all ngrams of length n starting from the node root
    """
    if root not in graph:
        return [] # why would this be empty?
    anchor = GramPair(graph[root], role, id=graph[root]["id"])
    if n == 1:
        return [[anchor]]
    elif not graph[root].get("roles", []):
        return [[anchor]]
    res = []
    for edge in graph[root]["roles"]:
        if skip and edge in skip:
            continue
        ptr = graph[root]["roles"][edge]
        def _edgecomp(p):
            if type(p) == dict:
                p = p["target"]
            if p[0] == "#":
                for g in ngrams_from_node_trips(graph, p[1:], role=edge, n=n-1, skip=skip):
                    # prevent loops
                    if graph[root]["id"] not in [x.id for x in g]:
                        # add to result
                        res.append([anchor]+g)
        if edge in ["UNIT"]: # add non-id edges here
            res.append([anchor, GramPair(ptr, edge, id="%s_%s" % (root, edge))])
        if type(ptr) == list:
            for x in ptr:
                _edgecomp(x)
        else:
            _edgecomp(ptr)

    return res

from pytrips.ontology import get_ontology
from pytrips.structures import TripsType

ont = get_ontology()

def coeff_edge(a, b):
    if a.label == b.label:
        return 1
    return 1

class NGramScorer:
    """
    General form of NGram scoring:
        Score(x, y) = gate_cond(x, y) * sum(sim(x, y) * coeff(xi,yi))
    """
    def __init__(self,
                 sim=node_sim(None),
                 coeff=lambda a, b: 1,
                 gate=lambda a, b: 1,
                 align="greedy",
                 skip=None
    ):
        self.sim = sim
        if coeff is None:
            self.coeff = lambda a, b: 1
        else:
            self.coeff = coeff
        if gate is None:
            self.gate = lambda a, b: 1
        else:
            self.gate = gate

        self._align = align
        self._skip = skip or []

    def score_ngram(self, ngram_1, ngram_2):
        """
        Takes a list of elements and returns an ngram score
        """
        if len(ngram_1) != len(ngram_2):
            return 0
        gate = 1
        score = 0
        for ai, bi in zip(ngram_1, ngram_2):
            gate *= self.gate(ai.node_(), bi.node_())
            if gate == 0: # in case scoring is actually difficult?
                return 0
            score += self.coeff(ai.edge_(), bi.edge_()) + self.sim(ai.node_(), bi.node_())
        return gate * score/(len(ngram_1) * 2)

    def align(self, ngrams_1, ngrams_2, method=None, aligned=False):
        """
        Find the max score alignment between ngram pairs
        aligned -> True returns alignment iff method allows
            greedy, munkres

       """
        if method is None:
            method = self._align
        if method.startswith("greedy"):
            return self.greedy(ngrams_1, ngrams_2, aligned=aligned)
        elif method == "average":
            # non-symmetric
            return self.average(ngrams_1, ngrams_2, unbalanced=True)
        elif method == "max_average":
            return self.average(ngrams_1, ngrams_2)
        elif method == "reverse_max_average":
            # non-symmetric
            return self.average(ngrams_2, ngrams_1)
        elif method == "bidir_average":
            return (self.average(ngrams_1, ngrams_2) +
                    self.average(ngrams_2, ngrams_1))/2
        elif method == "hyp_average":
            return math.sqrt((self.average(ngrams_1, ngrams_2) ** 2 +
                    self.average(ngrams_2, ngrams_1) ** 2)/2)
        elif method == "munkres":
            return self.munkres(ngrams_1, ngrams_2, aligned=aligned)

    def average(self, ngrams_1, ngrams_2, unbalanced=False):
        s1 = len(ngrams_1)
        s2 = len(ngrams_2)
        if s2 == 0 or s1 == 0:
            # if there are no ngrams available for either graph, the total score will be 0
            return 0
        if unbalanced:
            s2 = max(len(ngrams_1), len(ngrams_2))
            s1 = min(len(ngrams_1), len(ngrams_2))
        # non-symmetric
        total = []
        for a in ngrams_1:
            total.append(max([0] + [self.score_ngram(a, b) for b in ngrams_2]))
        return sum([x for x in reversed(sorted(total))][:s1])/s2

    def greedy(self, ngrams_1, ngrams_2, aligned=False):
        """
        Determine an alignment by greedy approximation.

        Hypothesis: Greedy approximation is a good approximation as long as similarity
        is function is convex
        """
        hyps = []
        for i in range(len(ngrams_1)):
            for j in range(len(ngrams_2)):
                hyps.append((self.score_ngram(ngrams_1[i], ngrams_2[j]), i, j))
        hyps = sorted(hyps, key=lambda x: -x[0])
        taken1, taken2 = set(), set()
        res = []
        for s, h1, h2 in hyps:
            if h1 in taken1 or h2 in taken2:
                continue
            res.append([s, ngrams_1[h1], ngrams_2[h2]])
            taken1.add(h1)
            taken2.add(h2)
        if aligned:
            return res
        else:
            return max(sum([r[0] for r in res]), 1)/max(len(ngrams_1), len(ngrams_2), 1)

    def munkres(self, ngrams_1, ngrams_2, aligned=False, reverse=False):
        """
        Use the munkres algorithm to compute optimal matching.
        """
        def pad(inp, k, pad_value=0):
            return inp + [pad_value]*(k-len(inp))
        # Make the cost matrix
        ng1 = len(ngrams_1)
        if ng1 < len(ngrams_2):
            return self.munkres(ngrams_2, ngrams_1, aligned, reverse=not reverse)
        matrix = [pad([1 - self.score_ngram(ngrams_1[i], ngrams_2[j])
                       for j in range(len(ngrams_2))], ng1)
                  for i in range(len(ngrams_1))]
        m = Munkres()
        indices = m.compute(matrix)
        if aligned:
            ngrams_2 = pad(ngrams_2, ng1, None)
            if reverse:
                return [[1 - matrix[i][j], ngrams_2[j], ngrams_1[i]] for i, j in indices if matrix[i][j] < 1]
            return [[1 - matrix[i][j], ngrams_1[i], ngrams_2[j]] for i, j in indices if matrix[i][j] < 1]
        # Why is this not normalized when the others are?
        res = [1 - matrix[i][j] for i, j in indices if matrix[i][j] < 1]
        return sum(res)/len(ngrams_1)#max(len(res), 1)

    def bleu(self, ngrams_1, ngrams_2, n=3, weights=[], aligned=False):
        """
        BLEU = BrevityPenalty * exp(geometric-mean(log(pk)))
        pk = K-gram precision
        """
        if type(n) is list:
            weights = n
            n = len(weights)
        else:
            weights = [1/n for i in range(n)]

        #HARDCODED trips
        ngrams_1 = [ngrams(ngrams_1, i+1, "trips", skip=self._skip) for i in range(n)]
        ngrams_2 = [ngrams(ngrams_2, i+1, "trips", skip=self._skip) for i in range(n)]
        #print([len(ng_1) for ng_1 in ngrams_1], [len(ng_2) for ng_2 in ngrams_2])
        alignment = [self.align(
                        ng1,
                        ng2, aligned=aligned) for ng1, ng2 in zip(ngrams_1, ngrams_2)]
        if aligned:
            return alignment
        score = zip(weights, alignment)
        # TODO: Is this the right corner case treatement
        def err(a, b):
            if b == 0:
                return 0
            else:
                return a*math.log(b)
        score = sum([err(a, b) for a, b in score])
        bp = math.exp(min(1 - len(ngrams_1)/len(ngrams_2), 0))
        return bp * math.exp(score)

def compare_trips(f1, f2, max_ngrams=3, sim=None, coeff=coeff_edge, gate=None, strategy="greedy", aligned=False, skip=None):
    """
    sim: "wup", "cosine", "eq"
    coeff:
    gate: "span", "lex"
    align: "greedy", "average", "max_average", "bidir_average", "hyp_average", "munkres"
    """
    scorer = NGramScorer(node_sim(sim), coeff, gate, strategy, skip=skip)
    return scorer.bleu(f1, f2, n=max_ngrams, aligned=aligned)
