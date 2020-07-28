
import networkx as nx
import random
from ..compare import kronecker, weighted_ngram_score
from ..load import dump_stg, load_stg

def node_choice(G):
    return random.choice([n for n in G.nodes])

def edge_choice(G):
    return random.choice([n for n in G.edges])

def random_node_labels(G, labels=None, slot="id", unique=False):
    """Applies random labels to a graph
    :param slot: the slot to put the label in
    :param labels: If none, choose a number between 0 and |G.nodes|, otherwise sample randomly from labels
    :param unique: If true, use each possible label only once.
    """
    if unique and labels and len(labels) < len(G.nodes):
        raise ValueError("Requested unique labels but provided too few labels")
    if not labels:
        labels = [i for i in range(len(G.nodes))]
    if not unique:
        labels = [random.choice(labels) for i in labels]
    random.shuffle(labels)
    for n in G.nodes:
        G.nodes[n][slot] = str(labels.pop())
    return G


def random_edge_labels(G, labels=None, slot="role", unique=False):
    """
    Applies random labels to a graph
    :param slot: the slot to put the label in
    :param labels: If none, choose a number between 0 and |G.nodes|, otherwise sample randomly from labels
    :param unique: If true, use each possible label only once.
    """
    if unique and labels and len(labels) < len(G.edges):
        raise ValueError("Requested unique labels but provided too few labels")
    if not labels:
        labels = [i for i in range(len(G.edges))]
    if not unique:
        labels = [random.choice(labels) for i in labels]
    random.shuffle(labels)
    for n in G.edges:
        G.edges[n][slot] = str(labels.pop())
    return G

def __handle_labeller(labeller, i, unique, fn):
    if type(labeller) is ElementSpace:
        if unique:
            labeller = labeller.seq()
        else:
            labeller = labeller.sample(k=i)
    if type(labeller) is list:
        return lambda G: fn(G, labels=labeller)
    return labeller

def random_tree(i, edge_label=random_edge_labels, node_label=random_node_labels, unique=False):
    """
    generates a randomly labelled tree. If edge_label or node_label are lists,
    generate a function to apply labels randomly from that list instead

    :param i: number of nodes
    :param edge_label: a function to assign edge_labels
    :param node_label: a function to assign node_labels
    """
    edge_label = __handle_labeller(edge_label, i-1, unique, random_edge_labels)
    node_label = __handle_labeller(node_label, i, unique, random_node_labels)

    G = nx.DiGraph()
    G.add_node(0)
    for n in range(1, i):
        G.add_edge(random.randrange(0, n), n)
    # Apply random labels to nodes and edges
    G = node_label(G)
    G = edge_label(G)
    return G

def is_connected(G):
    """
    naive: there is a node that can reach all others
    """
    for a, b in nx.all_pairs_shortest_path_length(G):
        if len(b) == len(G.nodes):
            return True
    return False

class MutationStrategy:
    def __init__(self, probabilities=None, thresholds=None):
        """
        with thresholds: NLABEL, ELABEL
        without:         NADD, EADD
        """
        if not probabilities:
            probabilities = dict(NLABEL=1, ELABEL=1, EADD=1, NADD=1)
        if not thresholds:
            thresholds = dict()
            if not "ELABEL" in thresholds:
                thresholds["ELABEL"] = 0.0
            if not "NLABEL" in thresholds:
                thresholds["NLABEL"] = 0.5
        self.keys, self.probabilities = list(zip(*[(k,v) for k,v in probabilities.items()]))
        self.thresholds = thresholds

    def next(self):
        res = random.choices(self.keys, weights=self.probabilities, k=1)[0]
        # print("Mutation:", res)
        return (res, self.thresholds.get(res, 0))

    @staticmethod
    def default():
        return MutationStrategy()

    @staticmethod
    def add():
        return MutationStrategy(probabilities=dict(NLABEL=0, ELABEL=0, EADD=1, NADD=1))

    @staticmethod
    def relabel(thresholds=None):
        return MutationStrategy(probabilities=dict(NLABEL=1, ELABEL=1, EADD=0, NADD=0))

class MutatedGraph:
    def __init__(self, G, history=None):
        if not history:
            history = []
        self.graph = G
        self.history = history

    def mutate(self, G, move):
        return MutatedGraph(G, [x for x in self.history] + [move])

    def __str__(self):
        """Add a comment with history and then dump_stg"""
        history = "#+HISTORY:"+" ".join(["%s|%f" % h for h in self.history])
        return "%s\n%s" % (history, dump_stg(self.graph))

    @staticmethod
    def load(string, edge_indicator="->", key_prefix=":"):
        lines = string.split("\n")
        history = []
        if lines[0] and lines[0][0].upper() == "#+HISTORY":
            history = [x.split("|") for x in lines[0][1:].split()]
            history = [(a, float(b)) for a, b in history]
        graph = load_stg(lines, edge_indicator, key_prefix)
        return MutatedGraph(graph, history)

class GraphSpace:
    def __init__(self, nodespace, edgespace, mutation=MutationStrategy(), node_label="id", edge_label="role"):
        self.nodespace = nodespace
        self.edgespace = edgespace
        self.node_label = node_label
        self.edge_label = edge_label
        self.mutation = mutation

    def weighted_ngram_score(self, edge_weight=1, node_weight=1):
        return weighted_ngram_score(edge=self.edgespace.similarity,
                                node=self.nodespace.similarity,
                                edge_weight=edge_weight, node_weight=node_weight)

    def choose(self, i, data=None):
        """Generate a random graph. Override this method to change graph generation"""
        # fixing unique to be true
        return MutatedGraph(random_tree(i, self.edgespace, self.nodespace, unique=True))

    def mutate(self, MG):
        """
        Mutation types:
            NLABEL - change a node label
            ELABEL - change an edge label

            # Structural mutations
            NADD - add a node
            EADD - add an edge

            # Potentially disconnecting mutations
            # hack - try the mutation 10 times and check to see that the digraph is still fully reachable
            NREMOVE1 - remove a node and reattach the edge
            NREMOVE2 - remove a node and the corresponding edge
            ETSWITCH - switch the targets of two edges
            ESSWITCH - switch the sources of two edges
            ESREATTACH - change the source of an edge
        """
        if not (type(MG) is MutatedGraph):
            MG = MutatedGraph(MG)
        G = MG.graph
        mut, thresh = self.mutation.next()
        if mut == "NLABEL":
            graph, move = self.nlabel(G, thresh)
            return MG.mutate(graph, ("NLABEL", move))
        if mut == "CADD":
            graph, move = self.cadd(G, thresh)
            return MG.mutate(graph, ("CADD", move))
        if mut == "ELABEL":
            graph, move = self.elabel(G, thresh)
            return MG.mutate(graph, ("ELABEL", move))
        if mut == "EADD":
            graph, move = self.eadd(G)
            return MG.mutate(graph, ("EADD", move))
        if mut == "NADD":
            graph, move = self.nadd(G)
            return MG.mutate(graph, ("NADD", move))
        return G

    def nlabel(self, G, threshold=0, node=None):
        if not node:
            node = node_choice(G)
        elt = G.nodes[node][self.node_label]
        new_label, dist = self.nodespace.mutate(elt, min_sim=threshold)
        if new_label == elt:
            return G.copy(), 0.0
        # print("NLABEL: %s -> %s" % (elt, new_label))
        G = G.copy()
        G.nodes[node][self.node_label] = new_label
        return G, dist

    def elabel(self, G, threshold=0, edge=None):
        if not edge:
            edge = edge_choice(G)
        elt = G.edges[edge][self.edge_label]
        new_label, dist = self.edgespace.mutate(elt, min_sim=threshold)
        if new_label == elt:
            return G.copy(), 0
        # print("ELABEL: %s[%s] -> %s" % (elt, str(edge), new_label))
        G = G.copy()
        G.edges[edge][self.edge_label] = new_label
        return G, dist

    def eadd(self, G, source=None, target=None, label=None):
        """
        actively avoids self-edges
        """
        if len(G.nodes) < 2:
            return G.copy(), 0
        if not source:
            source = node_choice(G)
            while source == target:
                source = node_choice(G)
        if not target:
            target = node_choice(G)
            while source == target:
                target = node_choice(G)
        if not label:
            label = self.edgespace.choose()
        # print("EADD: %s[%s,%s]" % (label, source, target))
        G = G.copy()
        G.add_edge(source, target, **{self.edge_label: label})
        return G, 1

    def nadd(self, G, source=None, label=None, edge=None):
        if not source:
            source = node_choice(G)
        if not label:
            label = self.nodespace.choose()
        if not edge:
            edge = self.edgespace.choose()
        i = 0
        while ("%s-%d" % (label, i)) in G.nodes:
            i += 1
        G = G.copy()
        v = "%s-%d" % (label, i)
        # print("NADD: %s[%s, +%s]" %(edge, source, v))
        G.add_edge(source, v, **{self.edge_label: edge})
        G.nodes[v][self.node_label] = label
        return G, 1

    def cadd(self, G, threshold=0, source=None, edge=None):
        if not source:
            source = node_choice(G)

        # mutate an existing node label within threshold to determine output label
        target = node_choice(G)
        elt = G.nodes[target][self.node_label]
        label, dist = self.nodespace.mutate(elt, min_sim=threshold)

        if not label:
            return G.copy(), 0
        if not edge:
            edge = self.edgespace.choose()
        i = 0
        while ("%s-%d" % (label, i)) in G.nodes:
            i += 1
        G = G.copy()
        v = "%s-%d" % (label, i)
        # print("NADD: %s[%s, +%s]" %(edge, source, v))
        G.add_edge(source, v, **{self.edge_label: edge})
        G.nodes[v][self.node_label] = label
        return G, dist

class ElementSpace:
    """
    Sample and mutate labels based off of probabilities. Similarity metric can be used to
    limit mutation distance
    """
    def __init__(self, labels, probabilities=None, similarity=None):
        if type(labels) is int:
            labels = [i for i in range(labels)]
        self.labels = labels
        if not probabilities or sum(probabilities) == 0:
            probabilities = [1] * len(labels)
        self.probabilities = [i for i in probabilities]
        if not similarity:
            similarity = kronecker
        self._similarity = similarity
        self._simcache = {}

    def size(self):
        return len(self.labels)

    def similarity(self, a, b):
        if (a, b) not in self._simcache:
            self._simcache[(a,b)] = self._similarity(a,b)
        return self._simcache[(a,b)]

    def sample(self, k=1):
        """
        Choose a random element from labels with replacement, weighted by probabilities
        """
        return random.choices(self.labels, weights=self.probabilities, k=k)

    def choose(self):
        return self.sample(k=1)[0]

    def seq(self):
        """
        return labels in a random order
        """
        labels = [x for x in self.labels]
        random.shuffle(labels)
        return labels

    def mutate(self, elt, min_sim=0):
        """
        Get a new label to replace elt whose similarity score is at least min_sim (with elt)
        """
        labels, probs = list(zip(*[(i, j) for i, j in zip(self.labels, self.probabilities)
                               if self.similarity(i, elt) >= min_sim]))
        if not labels:
            # There were no mutation candidates found
            return None, None
        res = random.choices(labels, weights=probs, k=1)[0]
        return res, self.similarity(elt, res)

def random_space(num_labels):
    labels = []
    for i in range(num_labels):
        cand = "".join([random.randint(0,1) for j in range(num_labels)])
        while cand in labels:
            cand = "".join([random.randint(0,1) for j in range(num_labels)])
        labels.append(cand)
    return labels

def space_from_tree(G):
    n = len(G.nodes)
    index = {k: i for i, k in enumerate(G.nodes)}
    vecs = []
    # choose root that has path to all nodes
    # generate vectors
    for x, paths in nx.all_pairs_shortest_path():
        if len(paths) != n:
            continue
        for p in paths.values():
            vecs.append([int(index[i] in p) for i in range(n)])
        return vecs

def str_vec_cosine(a, b):
    adotb = sum([int(ai) * int(bi) for ai, bi in zip(a, b)])
    A = math.sqrt(sum([int(ai) for ai in a]))
    B = math.sqrt(sum([int(bi) for bi in b]))
    return adotb/(A*B)
