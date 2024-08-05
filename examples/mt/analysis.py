from expkit import ExpSetup
from expkit.ops import (
    EvalMean,
    EvalLast,
    EvalMax,
    EvalTotalMean,
    EvalMeanLast,
    EvalMeanMax,
    Operation,
)
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re

# beta in used by the rlhf model: 0.0325
N = 1024
K = 64
TEMP = 0.6
N_MEASURAMENTS = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
]

eval_key = (
    "lastnumber"  # "crm:hamishivi-tulu-v2"
)
rm_path = "hamishivi/tulu-v2.5-7b-uf-rm"


setup = ExpSetup(
    "/gscratch/ark/graf/quest-rlhf/mt-outputs/",
    lazy=True,
    load_instances=False,
).query({"steps": 128})


# print(setup.meta())


base = setup.query(
    {
        "variant": "ancestral",
    }
)

quest = setup.query({"variant": "quest"})


for lp in ["en-zh", "en-cs"]:
    print("---" * 20)
    print(lp)
    print("base")
    print(
        len(
            base.query(
                {"language_pair": lp}
            ).meta()
        )
    )
    print(
        (
            [
                x["temperature"]
                for x in quest.query(
                    {"language_pair": lp}
                ).meta()
            ]
        )
    )
    print("quest")
    print(
        len(
            quest.query(
                {"language_pair": lp}
            ).meta()
        )
    )
    print(
        (
            [
                x["beta"]
                for x in quest.query(
                    {"language_pair": lp}
                ).meta()
            ]
        )
    )
