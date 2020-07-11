import networkx as nx
from tripsbleu.score import sembleu

def test_identity_trees():
    # generate some random trees
    # make sure they are equal to themselves
    for i in range(3, 100):
        for j in range(5):
            G = nx.random_tree(i)
            assert sembleu(G, G) == 1
