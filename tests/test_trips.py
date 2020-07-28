import json, os
from tripsbleu.load import tripsnx, dump_stg, load_stg, loads_stg
from tripsbleu.score import sembleu
from tripsbleu.compare import trips_ngram_score, greedy

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'data',
    )

JSON_FILES = os.path.join(FIXTURE_DIR, "json")
STG_FILES = os.path.join(FIXTURE_DIR, "stg")

def test_load_trips():
    for f in os.listdir(JSON_FILES):
        with open(os.path.join(JSON_FILES, f)) as data:
            G, root = tripsnx(json.load(data))
        assert G # basically assert that graph exists for now

        text = dump_stg(G)
        G2 = loads_stg(text)
        assert sembleu(G, G2) # written to STG and back is the same

        with open(os.path.join(STG_FILES, f[:-4] + "stg")) as data:
            G3 = load_stg(data)
        assert sembleu(G, G3) # fixed point

def test_compare_stg():
    for f in os.listdir(STG_FILES):
        with open(os.path.join(STG_FILES, f)) as data:
            G1 = load_stg(data)
        with open(os.path.join(STG_FILES+"2", f)) as data:
            G2 = load_stg(data)
        print(f)
        print(sembleu(G1, G2, node_label="type"))
        print("strict=True")
        print(sembleu(G1, G2, pk=greedy(score=trips_ngram_score(strictness=1)), node_label="type"))
        print("strict=False")
        print(sembleu(G1, G2, pk=greedy(score=trips_ngram_score(strictness=0)), node_label="type"))
    assert False
