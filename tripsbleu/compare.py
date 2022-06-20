from collections import defaultdict, Counter
from pytrips.ontology import get_ontology as ont
import Levenshtein


def levenshtein(x, y):
    return 1 - Levenshtein.distance(x, y)/max(len(x), len(y))

def jaro(x,y):
    # differ mostly in suffix
    return Levenshtein.jaro_winkler(x, y)

def stepenshtein(x, y):
    # Differ by like 1-2 characters
    if Levenshtein.distance(x, y) < 2:
        return 0.9
    return 0

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

def trips_wup(a, b):
    """
    Returns the wup score for a and b in trips
    if a or b are not trips elements, returns kronecker instead
    """
    a1 = ont()[a]
    b1 = ont()[b]
    if a1 and b1 and a1.parent and b1.parent:
        return a1.wup(b1)
    return kronecker(a, b)

def weighted_ngram_score(edge=kronecker, node=kronecker, edge_weight=1, node_weight=1, strictness=-1):
    """
    An ngram is a list of n node-labels interspersed with n-1 edge labels.
    This function returns a function to score a pair of ngrams
    :param edge: a function to score the agreement of edge labels
    :param node: a function to score the agreement of node labels
    :param edge_weight: How much the edge score counts towards the total
    :param node_weight: How much the node score counts towards the total
    :param strictness: if strictness >= 0, ngrams with an element having similarity <= strictness are assigned a score of 0
                       This behavior should better mimic sembleu when strictness approaches 1
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
        norm = 0
        for i in range(len(comp)):
            if i % 2 == 0:
                ns = node(*comp[i])
                if ns < strictness or (strictness == 0 == ns):
                    return 0
                score += ns * node_weight
                norm += node_weight
            else:
                es = edge(*comp[i])
                if es < strictness or (strictness == 0 == es):
                    return 0
                score += es * edge_weight
                norm += edge_weight
        return score / norm #(nodes * node_weight + edges * edge_weight)
    return _ng_comp


def trips_ngram_score(edge=kronecker, edge_weight=1, node_weight=1, strictness=0):
    return weighted_ngram_score(edge=edge, node=trips_wup, edge_weight=edge_weight, node_weight=1, strictness=strictness)

def greedy(score=weighted_ngram_score(), align=False):
    """
    :param score: an ngram scoring function
    :param align: whether or not the actual alignment should be returned
    :returns: a function that computes the greedy scoring between two lists of ngrams
    """
    def _greedy(source, reference):
        """
        :param source: A list of n-grams
        :param reference: A list of n-grams
        :returns: A score or alignment between the two lists
        """
        if (not source) or (not reference):
            return 0
        scored = []
        for i, x in enumerate(source):
            for j, y in enumerate(reference):
                scored.append((i, j, score(x, y)))
        taken_x, taken_y = set(), set()
        alignments = []
        res = 0
        while scored:
            i, j, s = max(scored, key=lambda x: x[2])
            alignments.append((source[i], reference[j], s))
            #print("%.3f: %s -> %s" % (s, source[i], reference[j]))
            res += s
            taken_x.add(i)
            taken_y.add(j)
            scored = [(i, j, s) for i, j, s in scored if i not in taken_x and j not in taken_y]
        if align:
            return res/len(reference), alignments
        return res/len(reference)
    return _greedy

def ngram_to_node_alignments(alignments):
    """
    Given a list of ngram-to-ngram alignments, return the estimated node-to-node alignments
    naive implementation: count the number of times a pair of nodes are aligned
    for each node in the reference, choose the most-aligned node in the candidate
    """
    res = defaultdict(Counter)
    for c1, c2, _s in alignments:
        for n1, n2 in zip(c1[::2], c2[::2]):
            res[n1][n2] += 1
    return {r: candidates.most_common()[0][0] for r, candidates in res.items()}
