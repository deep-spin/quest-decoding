from expkit import DiskStorage
from expkit import ExpSetup


lp = ["en-is", "en-zh", "en-cz"]
metric = ["qe:Unbabel-wmt23-cometkiwi-da-xl"]
setup = ExpSetup(
    DiskStorage("/gscratch/ark/graf/quest-decoding/examples/mt/mt-outputs/"),
).query(
    {
        "model_path": "haoranxu/ALMA-7B",
        "reward_model_path": "Unbabel/wmt23-cometkiwi-da-xl",
    }
)

lps = {m["language_pair"] for m in setup.meta()}

lp_count = {lp: len(setup.query({"language_pair": lp})) for lp in lps}

print(lp_count)


for e in setup.query({"language_pair": "en-is"}).sort("at"):

    # if e.get("at") >= "2024-08-22T14:06:26.102940":
    print("--" * 20)
    print(e.get("at"))
    print(e.get("variant"))
    print(e.evals().keys())
    # print(e.evals()["qe:Unbabel-wmt23-cometkiwi-da-xl"])

import pdb

pdb.set_trace()
