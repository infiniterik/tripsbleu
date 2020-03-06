from .ngrams import compare_trips, ngrams_trips, GramPair
from copy import deepcopy

def trips_diff(g1, g2, diff_only=False):
    """takes two graphs and colors nodes and edges red if they differ"""
    alignments = compare_trips(g1, g2, strategy="munkres", aligned=True)
    # 1. Take *all* aligned triples and construct a list of (Nd-eg-Nd, Nd-eg-Nd) pairs
    # 2. For all Nd-eg-Nd instances (by label) in the target graph
    #         color red if no tuple from above matches by value
    triples = alignments[1]
    # this is the kind of bs that makes my code hard to come back to
    # that, and no meaningful comments in algorithms
    edges = {tuple(x[1:]): False for x in GramPair.flatten(ngrams_trips(g2, n=2))}
    nodes = [b[1] for v, a, b in alignments[0] if v != 1.0]

    for v, s, t in triples:
        s = list_to_tuples(s)
        t = list_to_tuples(t)
        for a, b in zip(s, t):
            if a[1].value == b[1].value:
                edges[b] = True
    res = dict(edges=[(x[0].id, x[1].value, "#"+x[2].id) for x, v in edges.items() if not v], nodes=nodes)
    if diff_only:
        return res
    g2 = deepcopy(g2)
    for s, t, v in res["edges"]:
        for utt in g2:
            if s in utt and utt[s]["roles"][t] == v:
                utt[s]["roles"][t] = dict(style=dict(color="red"), target=v)
    return g2

def list_to_tuples(seq):
    collected = []
    curr = []
    state = 0
    for s in seq:
        # start pulling triples with a type
        if state == 0:
            if s.type == "type":
                curr.append(s)
                state += 1
        # second element should be an edge
        elif state == 1:
            if s.type == "edge":
                curr.append(s)
                state += 1
            # two nodes in a row, reset at the second node
            # shouldn't ever happen
            else:
                curr = [s]
        # end with another type
        elif state == 2:
            if s.type == "type":
                curr.append(s)
                collected.append(tuple(curr))
            curr = [s]
            state = 1
    return collected
