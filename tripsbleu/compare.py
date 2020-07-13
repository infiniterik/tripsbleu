def kronecker(x, y):
    """
    :returns: 1 if x == y, 0 otherwise
    """
    return int(x == y)

def l1(x, y, range_=(0,1)):
    """returns the the l1-similarity between 0 and 1 of two scalar values
    :param range: the range from which x and y are drawn
    :returns: a value between 0 and 1
    """
    if not (range[0] <= x <= range[1]):
        raise ValueError("value %d was outside of range %s" % (x, str(range_)))
    if not (range[0] <= y <= range[1]):
        raise ValueError("value %d was outside of range %s" % (y, str(range_)))

    return 1 - abs(x - y)/abs(range_[0] - range_[1])

def set_intersection(source, reference):
    """
    :returns: the number of elements of source that are in reference, divided by the size of reference
    """
    if (not source) or (not reference):
        return 0
    return len(set(source).intersection(set(reference)))/len(set(reference))

def weighted_ngram_score(edge=kronecker, node=kronecker, edge_weight=1, node_weight=1):
    """
    An ngram is a list of n node-labels interspersed with n-1 edge labels.
    This function returns a function to score a pair of ngrams
    :param edge: a function to score the agreement of edge labels
    :param node: a function to score the agreement of node labels
    :param edge_weight: How much the edge score counts towards the total
    :param node_weight: how much the node score counts towards the total
    :returns: An ngram_comparison function
    """
    def _ng_comp(source, reference):
        """
        :returns: a score between 0 and 1 for the ngram
        """
        if len(source) != len(reference):
            raise ValueError("ngrams %s and %s are of different lengths" % (source, reference))
        if len(source) % 2 == 0:
            raise ValueError("Got ngram of length %d, expected odd length" % len(source))
        comp = [_ for _ in zip(source, reference)]
        score = 0
        for i in range(len(comp)):
            if i % 2 == 0:
                score += node(*comp[i]) * node_weight
            else:
                score += edge(*comp[i]) * edge_weight
        edges = len(source)//2
        nodes = len(source) - edges
        return score / (nodes * node_weight + edges * edge_weight)
    return _ng_comp

def greedy(score=weighted_ngram_score(), align=False):
    """
    :param score: an ngram scoring function
    :param align: whether or not the actual alignment should be returned
    :returns: a function that computes the greedy scoring between two lists of ngrams
    """
    def _greedy(reference, candidate):
        scored = []
        for i, x in enumerate(reference):
            for j, y in enumerate(candidate):
                scored.append((i, j, score(x, y)))
        taken_x, taken_y = set(), set()
        alignments = []
        res = 0
        while scored:
            i, j, s = max(scored, key=lambda x: x[2])
            alignments.append((reference[i], candidate[j], i, j, s))
            res += s
            taken_x.add(i)
            taken_y.add(j)
            scored = [(i, j, s) for i, j, s in scored if i not in taken_x and j not in taken_y]
        if align:
            return res/len(candidate), aligned
        return res/len(candidate)
    return _greedy
