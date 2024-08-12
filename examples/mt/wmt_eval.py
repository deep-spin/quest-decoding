import os


from expkit import (
    ExpSetup,
    Exp,
    InstanceEval,
    Evalutor,
)

from expkit.ops import proj
from typing import *
from quest import (
    Reward,
)

from quest.reward.mt import CometModel

from quest.utils.list import (
    flatten_list,
    unflatten_list,
    get_unique_mapping,
    invert_unique_mapping,
)
from tqdm import tqdm
import numpy as np
import re

from sacrebleu.metrics import (
    BLEU,
    CHRF,
    TER,
)

import multiprocessing as mp


def sigmoid(x):
    x = np.array(x)
    return 1 / (1 + np.exp(-x))


def compress_predictions_mt(experiment):
    tokens, vocabs, counts = (
        [],
        [],
        [],
    )
    references, sources = [], []

    for i in tqdm(experiment.instances):

        tksi, vcbi = get_unique_mapping(
            list(
                map(
                    proj("text"),
                    i.outputs,
                )
            )
        )

        sourcesi = [
            i.input_data["source_sentence"]
        ] * len(vcbi)

        referencesi = [
            i.input_data[
                "reference_sentence"
            ]
        ] * len(vcbi)

        references.append(referencesi)
        sources.append(sourcesi)
        vocabs.append(vcbi)
        tokens.append(tksi)
        counts.append(len(vcbi))

    return (
        sources,
        references,
        tokens,
        vocabs,
        counts,
    )


class MTRewardEval(Evalutor):

    def __init__(self, reward: Reward):
        super().__init__(
            "corrected" + reward.get_name()
        )
        self.reward = reward

    def eval(
        self, experiment: Exp
    ) -> List[InstanceEval]:

        (
            sources,
            references,
            tokens,
            vocabs,
            counts,
        ) = compress_predictions_mt(
            experiment.refresh(force=True)
        )

        self.reward.set_sources(
            flatten_list(sources)
        )
        self.reward.set_references(
            flatten_list(references)
        )

        scores = sigmoid(
            self.reward.evaluate(
                candidates=list(
                    map(
                        lambda x: x.replace(
                            "</s>", ""
                        ),
                        flatten_list(
                            vocabs,
                        ),
                    )
                ),
                use_tqdm=True,
            )
        ).tolist()

        scores = unflatten_list(
            scores,
            counts,
        )

        duplicated_scores = [
            {
                "scores": invert_unique_mapping(
                    tks, rsi
                )
            }
            for tks, rsi in zip(
                tokens, scores
            )
        ]
        return duplicated_scores


class CorpusDiversityMetric:
    def __init__(
        self, name="", count_repeats=True
    ) -> None:
        super().__init__()
        self.name = name
        self.count_repeats = count_repeats

    def get_name(self):
        return self.name

    def compute_diversity(
        self, prediction_set
    ):
        pass

    def compute_batch(self, corpus):

        return list(
            map(
                self.compute_diversity,
                corpus,
            )
        )

    def __call__(self, corpus):
        return self.compute_batch(corpus)


class Pairwise(CorpusDiversityMetric):

    def __init__(
        self,
        name="bleu",
        metric=BLEU(),
        **kwargs,
    ) -> None:
        super().__init__(
            name=f"pairwise-{name}",
            **kwargs,
        )

        self.metric = metric

    """def corpus_eval(self, sys_stream, ref_streams):
        # bleu = _corpus_chrf(sys_stream, ref_streams, tokenize="none")

        bleu = self.metric.corpus_score(sys_stream, [ref_streams])
        return bleu.score"""

    def pairwise(
        self, sents
    ):  # non repeat sents.
        _ref, _hypo = [], []

        n = len(sents)

        if n > 1:
            d = np.zeros(
                (n, n), dtype=np.float32
            )
            for i in range(len(sents)):
                for j in range(len(sents)):
                    if i != j:
                        d[i, j] = 1 - (
                            self.metric.sentence_score(
                                sents[i],
                                [sents[j]],
                            ).score
                            / 100
                        )
                        # _ref.append(sents[i])
                        # _hypo.append(sents[j])

            return d
        else:
            return np.array([[0]])

    def compute_diversity(self, sentences):

        return self.pairwise(
            sentences
        ).tolist()

    def compute_batch(self, corpus):

        with mp.Pool() as p:
            return list(
                p.map(
                    self.compute_diversity,
                    corpus,
                )
            )

        # diversity_unique = list(map(self.compute_diversity, tqdm(corpus)))

        # return diversity_unique


class DiversityEval(Evalutor):

    def __init__(
        self, metric: CorpusDiversityMetric
    ):
        super().__init__(
            metric.get_name() + "-repr"
        )
        self.metric = metric

    def eval(
        self, experiment: Exp
    ) -> List[InstanceEval]:

        if (
            "quest"
            in experiment.meta["variant"]
        ):
            outputs = [
                [
                    o["text"]
                    for o in i.outputs
                    if o["accept"]
                ]
                for i in experiment.instances
            ]
        else:
            outputs = [
                list(
                    map(
                        proj("text"),
                        i.outputs,
                    )
                )
                for i in tqdm(
                    experiment.instances
                )
            ]

        scores = self.metric.compute_batch(
            outputs
        )

        evals = [
            {"scores": [s]} for s in scores
        ]

        return evals


class PairwiseBLEU(Pairwise):

    def __init__(
        self, unique=True, **kwargs
    ) -> None:
        super().__init__(
            name="bleu",
            metric=BLEU(
                effective_order=True
            ),
            **kwargs,
        )

        self.unique = unique

    def compute_diversity(self, sentences):

        if self.unique:
            sentences = list(set(sentences))

        return self.pairwise(
            sentences
        ).tolist()


def main(
    base_dir="mt-outputs/",
    reward_model_path="Unbabel/XCOMET-XL",
    batch_size=16,
    device_count=1,
    clamp: float = 1e-3,
):

    setup = ExpSetup(
        base_dir,
        lazy=True,
        load_instances=True,
    )

    if len(setup.experiments) == 0:
        raise FileNotFoundError(
            "The experiment has no data!"
        )

    if reward_model_path == "diversity":
        ps_eval = DiversityEval(
            PairwiseBLEU(unique=False)
        )

    else:
        reward = CometModel(
            model_path=reward_model_path,
            batch_size=batch_size,
            device_count=device_count,
            clamp=clamp,
        )

        ps_eval = MTRewardEval(
            reward=reward
        )

    setup = setup.filter(
        lambda x: not x.has_eval(
            ps_eval.eval_name
        )
    )

    print(setup.map(lambda x: x.name))

    setup = setup.safe_map(
        lambda experiment: (
            ps_eval(experiment)
            if not experiment.has_eval(
                ps_eval.eval_name
            )
            # and (
            #    experiment.meta["variant"]
            #    == "ancestral"
            # )
            else experiment
        )
    )

    # new_setup.save()


if __name__ == "__main__":

    import fire

    fire.Fire(main)
