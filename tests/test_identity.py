import networkx as nx
import random
from tripsbleu.score import sembleu, ngrams
from tripsbleu.load import loads_stg, dump_stg

def random_tree(i, edge_label="role", node_label="id"):
    G = nx.DiGraph()
    G.add_node(0)
    for n in range(1, i):
        G.add_edge(random.randrange(0,n), n)

    for k in G.nodes:
        G.nodes[k][node_label] = str(random.randrange(0,i))
    for e in G.edges:
        G.edges[e][edge_label] = str(random.randrange(0,i))
    return G

def test_identity_trees():
    # generate some random trees
    # make sure they are equal to themselves
    for i in range(3, 100):
        for j in range(5):
            G = random_tree(i)
            assert sembleu(G, G) == 1

def test_write_load():
    # generate some random trees
    # add ids and edge_labels
    for i in range(3, 100):
        for j in range(5):
            G1 = random_tree(i)
            text = dump_stg(G1)
            G2 = loads_stg(text)
            assert G1
            assert G2
            assert sembleu(G1, G2) == 1
