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

def add_trips_node(G, node, clone=True):
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
    for k, v in node.items():
        if k != "roles":
            G.nodes[id][k] = v
    for k, v in values.items():
        # Flattening the values dict
        if k in G.nodes[id]:
            raise KeyError("duplicate key %s when loading trips node %s" % (k, id))
        G.nodes[id][k] = v
    return G


def loads_stg(data, edge_indicator="->", key_prefix=":"):
    data=data.split("\n")
    return load_stg(data, edge_indicator, key_prefix)

def load_stg(data, edge_indicator="->", key_prefix=":"):
    """
    Node:
    id :key value :key2 value2
    Edge:
    id -> id2 :key value :key2 value2
    """
    def process_keys(keys, i):
        if len(keys) % 2 == 1:
            raise ValueError("Line %d: Key-Value sequence has an odd number of elements" % i)
        res = dict()
        for j in range(len(keys)//2):
            if not keys[2*j].startswith(key_prefix):
                raise ValueError("Line %d: Expected key (starting with %s), got %s instead" % (i, key_prefix, keys[2*j]))
            res[keys[2*j][len(key_prefix):]] = keys[2*j+1]
        return res

    G = nx.DiGraph()
    for i, line in enumerate(data):
        if not line or line.startswith("#"):
            # skip empty or commented lines
            continue
        line = line.strip().split()
        if edge_indicator in line:
            # Sanity check
            if len(line) < 3:
                raise ValueError("Line %d: Expected node -> node :key value ..." % i)
            if line.count(edge_indicator) != 1:
                raise ValueError("Line %d: Found more than one EDGE_INDICATOR" % i)
            if line.index(edge_indicator) != 1:
                raise ValueError("Line %d: Expected EDGE_INDICATOR, found %s instead" % (i, line[1]))
            # Add an edge
            source = line[0]
            target = line[2]
            keys = process_keys(line[3:], i)
            G.add_edge(source, target, **keys)
        else: # else we at a node
            G.add_node(line[0])
            for k, v in process_keys(line[1:], i).items():
                G.nodes[line[0]][k] = v
    return G

def dump_stg(G, edge_indicator="->", key_prefix=":"):
    """
    writes the graph to STG format. keys with None values are discarded
    """
    res = ""
    res += "# Nodes\n"
    for n in G.nodes:
        keyvals = [(k, v) for k, v in G.nodes[n].items()]
        for k, v in keyvals:
            if v and type(v) not in [str, int, float, bool]:
                raise TypeError("Label value in node %s is of type %s which is not atomic" % (n, type(v).__name__))
        res += "%s %s\n" % (n, " ".join(["%s%s %s" % (key_prefix, k, v) for k, v in keyvals if v is not None]))
    res += "# Edges\n"
    for s, t in G.edges:
        keyvals = [(k, v) for k, v in G.edges[(s, t)].items()]
        for k, v in keyvals:
            if v and type(v) not in [str, int, float, bool]:
                raise TypeError("Label value in edge %s->%s is of type %s which is not atomic" % (s, t, type(v).__name__))
        res += "%s %s %s %s\n" % (s, edge_indicator, t, " ".join(["%s%s %s" % (key_prefix, k, v) for k, v in keyvals if v is not None]))
    return res

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
        roots.append(subtree.get("root", "")[1:])
        for x, v in subtree.items():
            if x == "root":
                continue
            add_trips_node(G, v, clone=False)
    return G, [r for r in roots if r]




import penman
from penman.models.amr import model

def load_amr(graph):
    g = nx.DiGraph()
    for s, l, t in graph.instances():
        g.add_node(s)
        g.nodes[s]["id"] = t
    for s, l, t in graph.triples:
        if l == ":instance":
            continue
        s,l,t = penman.models.amr.model.canonicalize((s,l,t))
        l = penman.models.amr.model.canonicalize_role(l)
        g.add_edge(u_of_edge=s, v_of_edge=t, label=l)
    return g

def load_amr_file(fname):
    return [load_amr(x) for x in penman.load(source=fname)]
