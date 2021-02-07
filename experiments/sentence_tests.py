#!/usr/bin/env python3

from tripsbleu.load import load_amr_file
from tripsbleu import load, score, compare
from .generate import smatch
import pandas as pd


edge = compare.kronecker
node = compare.jaro
greedy = compare.greedy(compare.weighted_ngram_score(edge=edge, node=node, edge_weight=1, node_weight=1, strictness=0.8))

ngram_size=5
weights=[0.1,0.4,0.3,0.1,0.1]

sbl = lambda a, b: score.sembleu(candidate=a, reference=b, weights=weights, n=ngram_size, node_label="id", edge_label="label", pk=compare.set_intersection)
tbl = lambda a, b: score.sembleu(candidate=a, reference=b, weights=weights, n=ngram_size, node_label="id", edge_label="label", pk=greedy)
smt = lambda a, b: smatch(a=b, b=a, edge_label="label")

graphs = [load_amr_file("sembleu_data/e86_test.ans%d"%(i+1)) for i in range(0,4)]
refs = load_amr_file("sembleu_data/0.e86_test.ref_filtered")

human = pd.read_csv("sembleu_data/amr_human.csv", index_col="NO.")

keys1 = ['A1-1_2', 'A2-1_2', 'A3-1_2']
keys2 = ['A1-3_4', 'A2-3_4', 'A3-3_4']

def system(graphs, refs, key, tbs):
    one, two = key[0].split("-")[-1].split("_")
    one = int(one)
    two = int(two)
    res = []
    for i, w in enumerate(human[key[0]]):
        g1s = tbs(graphs[one-1][i], refs[i])
        g2s = tbs(graphs[two-1][i], refs[i])
        if g1s > g2s:
            res.append(one)
        elif g1s < g2s:
            res.append(two)
        else:
            res.append("-")
    return res


def count_results(dec, sb):
    res = 0
    for i, a, b, c in sb.itertuples():
        if dec[i-1] == "-" or [a,b,c].count(dec[i-1]) > 1:
            res += 1
    return res


def run():
    dec_sb = [system(graphs, refs, keys1, sbl), system(graphs, refs, keys2, sbl)]
    dec_sb_count = [count_results(dec_sb[0], human[keys1]), count_results(dec_sb[1], human[keys2])]

    print("SemBleu:", dec_sb_count, sum(dec_sb_count)/200)

    dec_tb = [system(graphs, refs, keys1, tbl), system(graphs, refs, keys2, tbl)]
    dec_tb_count = [count_results(dec_tb[0], human[keys1]), count_results(dec_tb[1], human[keys2])]

    print("TripsBleu:", dec_tb_count, sum(dec_tb_count)/200)

    #dec_sm = [system(graphs, refs, keys1, smt), system(graphs, refs, keys2, smt)]
    #dec_sm_count = [count_results(dec_sm[0], human[keys1]), count_results(dec_sm[1], human[keys2])]

    #print("Smatch:", dec_sm_count, sum(dec_sm_count)/200)
