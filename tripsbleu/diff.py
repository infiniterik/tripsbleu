from .ngrams import compare_trips, ngrams_trips, GramPair
from copy import deepcopy

def trips_diff(g1, g2, diff_only=False, edge_color="red", node_color="pink", skip=None, strategy="munkres"):
    """takes two graphs and colors nodes and edgee red if they differ"""
    alignments = compare_trips(g1, g2, strategy=strategy, aligned=True, skip=skip)
    # 1. Take *all* aligned triples and construct a list of (Nd-eg-Nd, Nd-eg-Nd) pairs
    # 2. For all Nd-eg-Nd instances (by label) in the target graph
    #         color red if no tuple from above matches by label
    triples = alignments[1]
    # this is the kind of bs that makes my code hard to come back to
    # that, and no meaningful comments in algorithms
    edges = {tuple(x[1:]): False for x in GramPair.flatten(ngrams_trips(g2, n=2))}
    nodes = [b[1] for v, a, b in alignments[0] if v != 1.0]

    for v, s_, t_ in triples:
        print(s_, t_)
        s = list_to_tuples(sum([s_1._flatten() for s_1 in s_], []))
        t = list_to_tuples(sum([t_1._flatten() for t_1 in t_], []))
        for a, b in zip(s, t):
            if a[1].label == b[1].label: #and a[0].label == b[0].label and a[2].label == b[2].label:
                edges[b] = True
    res = dict(edges=[(x[0].id, x[1].label, "#"+x[2].id) for x, v in edges.items() if len(x) == 3 and not v],
        nodes=nodes)
    if diff_only:
        return res
    g2 = deepcopy(g2)

    for s, t, v in res["edges"]:
        print(s, t, v)
        for utt in g2:
            if s in utt:
                target = utt[s]["roles"][t]
                if type(target) is not str:
                    style = target.get("style", {})
                    color = style.get("color", "") + ":"
                    target = target["target"]
                else:
                    color = ""
                if target == v:
                    color += edge_color
                    utt[s]["roles"][t] = dict(style=dict(color=color), target=v)
    for s in nodes:
        for utt in g2:
            if s.id in utt:
                color = utt[s.id].get("style", {}).get("fillcolor", "")
                if color:
                    color += ":%s" % node_color
                    print("alt color")
                    style = "striped"
                else:
                    color = node_color
                    style = "filled"
                utt[s.id]["style"] = dict(fillcolor=color, style=style)

    return g2

def three_way_diff(source, first, second, skip=None):
    source_second = trips_diff(source, second, edge_color="blue", node_color="lightblue", skip=skip)
    source_first = trips_diff(source, first, edge_color="red", node_color="pink", skip=skip)
    gold_first = trips_diff(first, source, edge_color="red", node_color="pink", skip=skip)
    source_both = trips_diff(second, gold_first, edge_color="blue", node_color="lightblue", skip=skip)
    return source_first, source_second, source_both

def list_to_tuples(seq):
    collected = []
    curr = []
    state = 0
    if not seq:
        return collected
    for s in seq:
        # start pulling triples with a type
        if s is None:
            return collected
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
