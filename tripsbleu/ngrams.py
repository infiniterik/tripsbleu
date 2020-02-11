from pprint import pprint
import math
import json
from munkres import Munkres

class GramPair:
    """
    A gram is a combination of a node and an incoming edge.
    the default incoming edge is "ROOT" and can be ignored
    """
    def __init__(self, node, edge="ROOT", graph_type="trips"):
        self.node = node
        self.edge = edge
        self.graph_type = graph_type

    def __repr__(self):
        return "%s->%s" % (self.edge, self.node["id"])

    def _flatten(self, node_type="type"):
        return [GramElement(self.edge, "edge"), GramElement(self.node, node_type)]

    @staticmethod
    def flatten(lst, node_type="type", edge_type="edge"):
        return [sum([x._flatten(node_type) for x in y], []) for y in lst]

class GramElement:
    """
    Type asserting wrapper for labels in NGrams.
    Edges and Nodes are treated differently.  May even want a mechanism for comparing different
    types of objects
    """
    def __init__(self, value, type_, extract=None):
        """
        """
        self.value = value
        self.type = type_
        self.extract = extract

    @property
    def label(self):
        if self.extract:
            return self.extract(self.value)
        if self.type == "edge":
            return self.value
        if self.type == "type":
            return self.value["type"]
        if self.type == "word":
            return self.value["word"]
        if self.type == "lex":
            return self.value["roles"]["LEX"]
        return None

    def __repr__(self):
        return "GE({}.{})".format(self.label, self.type)

    def __eq__(self, o):
        if isinstance(o, GramElement):
            return self.type == o.type and self.label == o.label and self.label and o.label
        return False

def _node_sim(a, b, func="wup"):
    """
    Compute the score between to trips "type" GramElements
    """
    if (a.label == "ROOT") or (b.label == "ROOT") or (a.type != "type") or (b.type != "type"):
        return a.label == b.label
    a = ont[a.label]
    b = ont[b.label]

    if func == "wup":
        return a.wup(b)
    elif func == "cosine":
        return a.cosine(b)
    elif func == "eq" or func is None:
        return float(a == b)

def node_sim(func="wup"):
    """
    func types are "wup", "cosine", and "eq"
    """
    return lambda a, b: _node_sim(a, b, func=func)


def ngrams(graph, n=1, graph_type="trips"):
    """
    An ngram is defined as a path containing n nodes
    graph: a graph object for the corresponding type
    n: the number of nodes to include 
    """
    if graph_type == "trips":
        return ngrams_trips(graph, n=n)
    else:
        return []

def ngrams_trips(graph, n=1):
    """
    Turn a trips graph into a list of ngrams of length n.
    input is json_format["parse"]
    """
    if type(graph) is list:
        # flatten the graph and ignore root keys
        new_graph = {}
        for g in graph:
            for x, y in g.items():
                new_graph[x] = y
        return ngrams_trips(new_graph, n=n)
    else:
        # we are ignoring the root key and starting from each node.
        res = [ngrams_from_node_trips(graph, x, role="ROOT", n=n) for x in graph if x != "root"]
        return sum([r for r in res if r], []) # get rid of empties

def ngrams_from_node_trips(graph, root, role="ROOT", n=1):
    """
    recursively find all ngrams of length n starting from the node root
    """
    anchor = GramPair(graph[root], role)
    if n == 1:
        return [[anchor]]
    elif not graph[root].get("roles", []):
        return [[anchor]]
    res = []
    for edge in graph[root]["roles"]:
        ptr = graph[root]["roles"][edge]
        if ptr[0] != "#":
            continue
        for g in ngrams_from_node_trips(graph, ptr[1:], role=edge, n=n-1):
            if graph[root]["id"] not in [x.node['id'] for x in g]: # prevent loops
                # add to result
                res.append([anchor]+g)
    return res

from pytrips.ontology import get_ontology
from pytrips.structures import TripsType

ont = get_ontology()


class NGramScorer:
    """
    General form of NGram scoring:
        Score(x, y) = gate_cond(x, y) * sum(sim(x, y) * coeff(xi,yi))
    """
    def __init__(self,
                 sim=node_sim(None),
                 coeff=lambda a, b: 1,
                 gate=lambda a, b: 1,
                 align="greedy"):
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

    def score_ngram(self, ngram_1, ngram_2):
        """
        Takes a list of elements and returns an ngram score
        """
        if len(ngram_1) != len(ngram_2):
            return 0
        gate = 1
        score = 0
        for ai, bi in zip(ngram_1, ngram_2):
            gate *= self.gate(ai, bi)
            if gate == 0: # in case scoring is actually difficult?
                return 0
            score += self.coeff(ai, bi) * self.sim(ai, bi)
        return gate * score/len(ngram_1)

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
            return sqrt((self.average(ngrams_1, ngrams_2) ** 2 +
                    self.average(ngrams_2, ngrams_1) ** 2)/2)
        elif method == "munkres":
            self.munkres(ngrams_1, ngrams_2, aligned=aligned)

    def average(self, ngrams_1, ngrams_2, unbalanced=False):
        s1 = len(ngrams_1)
        s2 = len(ngrams_1)
        if unbalanced:
            s2 = max(len(ngrams_1), len(ngrams_2))
            s1 = min(len(ngrams_1), len(ngrams_2))
        # non-symmetric
        total = []
        for a in ngrams_1:
            total.append(max([self.score_ngrams(a, b) for b in ngrams_2]))
        return reversed(sorted(total))[:s1]/s2

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
            return sum([r[0] for r in res])/len(res)

    def munkres(self, ngrams_1, ngrams_2, aligned=False):
        """
        Use the munkres algorithm to compute optimal matching.
        """
        def pad(inp, l):
            if len(inp) < l:
                return inp + [0]*(l-len(inp))
            return inp
        # Make the cost matrix
        ng1 = len(ngrams_1)
        if ng1 < len(ngrams_2):
            return munkres(ngrams_2, ngrams_1, aligned)
        matrix = [pad([1 - self.score_ngram(ngrams_1[i], ngrams_2[j])
                       for j in range(len(ngrams_2))], ng1)
                  for i in range(len(ngrams_1))]
        m = Munkres()
        indices = m.compute(matrix)
        if aligned:
            return [[1 - matrix[i][j], ngrams_1[i], ngrams_2[j]] for i, j in indices]
        # Why is this not normalized when the others are?
        return sum([1 - matrix[i][j] for i, j in indices])

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
        ngrams_1 = [ngrams(ngrams_1, i+1, "trips") for i in range(n)]
        ngrams_2 = [ngrams(ngrams_2, i+1, "trips") for i in range(n)]
        alignment = [self.align(
                        GramPair.flatten(ng1),
                        GramPair.flatten(ng2), aligned=aligned) for ng1, ng2 in zip(ngrams_1, ngrams_2)]
        if aligned:
            return alignment
        score = zip(weights, alignment)
        score = sum([a*math.log(b) for a, b in score])
        bp = math.exp(min(1 - len(ngrams_1)/len(ngrams_2), 0))
        return bp * math.exp(score)

def compare_trips(f1, f2, max_ngrams=3, sim=None, coeff=None, gate=None, align="greedy"):
    """
    sim: "wup", "cosine", "eq"
    coeff:
    gate: "span", "lex"
    align: "greedy", "average", "max_average", "bidir_average", "hyp_average", "munkres"
    """
    scorer = NGramScorer(node_sim(sim), coeff, gate, align)
    f1 = json.load(open(f1))["parse"]
    f2 = json.load(open(f2))
    if f2.get("alternatives"):
        f2 = f2["alternatives"][1]
    else:
        raise ValueError("Alternatives not found")
        f2 = f2["parse"]
    #pprint(scorer.bleu(f1, f2, n=3, aligned=True))

    return scorer.bleu(f1, f2, n=max_ngrams, aligned=False)

def compare_sentence(sentence, max_ngrams=3, sim=None, align="greedy"):
    """
    compare a parse against its first alternative
    """
    from tripscli.parse.web.parse import TripsParser
    parser = TripsParser(url="http://trips.ihmc.us/parser/cgi/step", debug=False)
    scorer = NGramScorer(sim=node_sim(sim), align=align)
    parser.set_parameter("number-parses-desired", 2)
    js = parser.query(sentence)
    f1 = js["parse"]
    f2 = js.get("alternatives")[0]
    pprint([[y for y in x if y[0] != 1.0] for x in scorer.bleu(f1, f2, n=max_ngrams, aligned=True)])
    return scorer.bleu(f1, f2, n=max_ngrams, aligned=False)
