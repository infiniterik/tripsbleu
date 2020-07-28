from tripsbleu.tools.generate import *
from tripsbleu.compare import trips_wup, kronecker
from tripsbleu.load import dump_stg
from pytrips.ontology import get_ontology as ont

from tripsbleu.score import sembleu
from tripsbleu.compare import trips_ngram_score, greedy

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

import smatch as _smatch
import penman

def _triples(g, edge_label="role"):
    t = []
    for e in g.edges:
        t.append(("v"+str(e[0]), g.edges[e][edge_label], "v"+str(e[1])))
    return t

def _instances(g, node_label="id"):
    i = []
    for n in g.nodes:
        i.append(("v"+str(n), ":instance", g.nodes[n].get(node_label, n)))
    return i

def nx_to_amr(g, node_label="id", edge_label="role"):
    return penman.Graph(triples=set(_triples(g, edge_label=edge_label)+_instances(g, node_label=node_label)))

def smatch(a, b, node_label="id", edge_label="role"):
    ga = penman.encode(g=nx_to_amr(a, node_label, edge_label), compact=False)
    gb = penman.encode(g=nx_to_amr(b, node_label, edge_label), compact=False)
    x1, x2, x3 = _smatch.get_amr_match(ga, gb)
    return _smatch.compute_f(x1, x2, x3)[-1]

STRATEGY = dict(
    DEFAULT=MutationStrategy.default(),
    ADD=MutationStrategy.add(),
    RELABEL=MutationStrategy.relabel(),
    RELABEL2=MutationStrategy.relabel(dict(NLABEL=0.8)),
    CADD=MutationStrategy(probabilities=dict(CADD=1, EADD=1), thresholds=dict(CADD=0.8)),
    RELABEL3=MutationStrategy.relabel(dict(NLABEL=0.3))
)

def random_graph_generator(strategy=None):
    """
    1. Create an element space (edge/node spaces)
    2. Create the graph space with a mutation strategy
    3. Generate a tree
    4. Apply mutations to the tree
    5. Observe changes
    """
    node_labels = [str(x) for x in ont()._data.keys()]
    edge_labels_core = "AGENT,AFFECTED,AFFECTED-RESULT,NEUTRAL,EXPERIENCER,FORMAL,BENEFICIARY"
    edge_labels_rels = "RESULT,SOURCE,TRANSIENT-RESULT,METHOD,REASON,MANNER"
    edge_labels_locs = "LOCATION,TIME,EXTENT,ORIENTATION,FREQUENCY"
    edge_labels_mods = "MOD,ASSOC-WITH,ASSOC-POS,IDENTIFIED-AS"
    edge_labels_advs = "FIGURE,GROUND,SCALE,STANDARD,EXTENT,DEGREE"
    all_edge_labels = ",".join([
        edge_labels_core,
        edge_labels_rels,
        edge_labels_locs,
        edge_labels_mods,
        edge_labels_advs
    ]).split(",")

    nodespace = ElementSpace(node_labels, similarity=trips_wup)
    edgespace = ElementSpace(all_edge_labels, similarity=kronecker)
    if not strategy:
        strategy = MutationStrategy.default()
    TRIPS = GraphSpace(nodespace, edgespace, mutation=strategy)
    return TRIPS

# full tripsbleu
tbleu0 = lambda G1, G2: sembleu(G1, G2, pk=greedy(score=trips_ngram_score(strictness=0)), node_label="id")
# full sembleu
tbleu1 = lambda G1, G2: sembleu(G1, G2, pk=greedy(score=trips_ngram_score(strictness=1)), node_label="id")

def simple_test(n=25, k=13, gr=None, verbose=True):
    if not gr:
        gr = random_graph_generator()
    base = gr.choose(k)
    return perturb(gr, base, n, verbose)

def triangle(n=5, k=13, verbose=True):
    gr = random_graph_generator()
    base = gr.choose(k)
    left = perturb(gr, base, n, verbose)
    right = perturb(gr, base, n, verbose)
    # do some processing here
    left_gr = left[2][-1]
    right_gr= right[2][-1]

    print("LR-Sem", tbleu1(left_gr.graph, right_gr.graph))
    print("BR-Sem", tbleu1(base.graph, right_gr.graph))
    print("BL-Sem", tbleu1(base.graph, left_gr.graph))
    print()
    print("LR-TBl", tbleu0(left_gr.graph, right_gr.graph))
    print("BR-TBl", tbleu0(base.graph, right_gr.graph))
    print("BL-TBl", tbleu0(base.graph, left_gr.graph))
    return left, right

def perturb(gr, base, n=25, verbose=True):
    df = pd.DataFrame(columns=["mutation", "inc", "tbleu0-increment", "tbleu1-increment", "tbleu0-base", "tbleu1-base", "smatch-increment", "smatch-base", "smatch-rev-base"])
    G1 = base
    graphs = [G1]
    i = 0
    while i < n:
        G2 = gr.mutate(G1)
        mut = G2.history[-1]
        graphs.append(G2)
        try:
            tb1inc = 1-tbleu1(G1.graph, G2.graph)
        except:
            tb1inc = 0
        try:
            tb0inc = 1-tbleu0(G1.graph, G2.graph)
        except:
            tb0inc = 0
        try:
            tb1base = tbleu1(base.graph, G2.graph)
        except:
            tb1base = 0
        try:
            tb0base = tbleu0(base.graph, G2.graph)
        except:
            tb0base = 0

        smb = smatch(base.graph, G2.graph)
        smrb = smatch(G2.graph, base.graph)
        smi = 1-smatch(G1.graph, G2.graph)

        if verbose:
            print("%2d: %s\n\tincrement%.3f -> %.3f" % (i+1, mut, tb1inc, tb0inc))
            print("\tbase     %.3f -> %.3f" % (tb1base, tb0base))
        df = df.append({
            "mutation": mut[0],
            "inc":mut[1],
            "tbleu1-increment":tb1inc,
            "tbleu0-increment":tb0inc,
            "tbleu1-base":tb1base,
            "tbleu0-base":tb0base,
            "smatch-increment":smi,
            "smatch-base":smb,
            "smatch-rev-base":smrb
        }, ignore_index=True)
        G1 = G2
        i += 1
    return df, gr, graphs
