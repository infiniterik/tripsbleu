import json, os
from tripsbleu.load import tripsnx, dump_stg, load_stg, loads_stg
from tripsbleu.score import sembleu


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
