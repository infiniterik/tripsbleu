import networkx as nx
import random
from tripsbleu.score import sembleu, ngrams
from tripsbleu.load import loads_stg, dump_stg
from tripsbleu.tools.generate import random_tree, ElementSpace

def test_identity_trees():
    # generate some random trees
    # make sure they are equal to themselves
    for i in range(3, 100):
        for j in range(5):
            G = random_tree(i)
            assert sembleu(G, G) == 1

def test_uniform_element_space():
    """
    This is a random test. It may fail occassionally
    """
    for i in range(1, 20):
        e = ElementSpace([j for j in range(i)], probabilities=[1 for j in range(i)])
        assert abs(sum(e.sample(100))/100 - (i-1)/2) < i/10

def test_average_element_space():
    for i in range(1, 20):
        probs = [random.betavariate(3,2)  for j in range(i)]
        j = sum(probs)
        probs = [k/j for k in probs]
        labels = [j for j in range(i)]
        expectation = sum([a*b for a, b in enumerate(probs)])
        e = ElementSpace(labels, probabilities=probs)
        assert abs(sum(e.sample(100))/100 - expectation) < i/10

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
