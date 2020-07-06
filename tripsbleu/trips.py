
def _node_sim(a, b, func="wup"):
    """
    Compute the score between to trips "type" GramElements
    """
    a = ont[a.label]
    b = ont[b.label]

    if func == "wup":
        return a.wup(b)
    elif func == "cosine":
        return a.cosine(b)
    elif func == "eq" or func is None:
        return a == b

def node_sim(func="wup"):
    """
    func types are "wup", "cosine", and "eq"
    """
    return lambda a, b: _node_sim(a, b, func=func)

def ngrams_trips(graph, n=1):
    """
    Turn a trips graph into a list of ngrams of length n.
    input is json_format["parse"]
    """
    if type(graph) is list:
        return sum([ngrams_trips(x, n=n) for x in graph], [])
    else:
        res = [ngrams_from_node_trips(graph, x, role="ROOT", n=n) for x in graph if x != "root"]
        return sum([r for r in res if r], []) # get rid of empties


# TODO: These should probably be somewhere in a parse module for trips
def get_nth_parse(parse, n, target_list="alternatives"):
    """Added a target list in case I want to store parses in places other than alternatives"""
    if target_list != "alternatives" and parse.get(target_list):
        if n < len(parse.get(target_list)):
            res = parse.get(target_list)[n]
            # deal with the weird case where the reparsed alts are a list of length 1, containing a parse
            if len(res) == 1 and type(res[0]) == list:
                return res[0]
            return res
        else:
            return None
    if n == 0:
        return parse.get("parse")
    elif n > 0:
        res = parse.get("alternatives")[n-1]
        if len(res) == 1 and type(res[0]) == list:
            return res[0]
        return res
    else:
        return None

def get_gold(parse, index=False):
    if "annotation" not in parse:
        return None
    else:
        ind =  parse["annotation"].get("judgement", -1)
        if index:
            return ind
        return get_nth_parse(parse, ind)
