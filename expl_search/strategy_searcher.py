import sys
sys.path.append('.')

import os
import argparse
import json
import re
import random

import numpy as np
import math

from tqdm import tqdm
from abc import ABC
from collections import namedtuple, Counter, OrderedDict
from os.path import join

import itertools
from functools import reduce

import heapq
from expl_search.inspections import inspection_print


class BufferedHeap:
    def __init__(self, buffer_size=4096):
        self.buffer_size = buffer_size
        self.heap = []
    
    # score must be the larger the better
    def push(self, comb, score, flag_tuple_score=False):
        if flag_tuple_score:            
            neg_score = tuple(-x for x in score)
            heapq.heappush(self.heap, (neg_score, comb))
        else:
            heapq.heappush(self.heap, (-score, comb))
        if len(self.heap) == 4 * self.buffer_size:
            self.consolidate()

    def consolidate(self):
        firstk = [heapq.heappop(self.heap) for _ in range(self.buffer_size)]
        self.heap = firstk
        heapq.heapify(self.heap)

    def get_topk(self, num):
        num = min(num, len(self.heap))
        return [heapq.heappop(self.heap) for _ in range(num)]    


def calc_coherence_score_of_a_combination(args, partial_idx, pair_idx_to_score):
    num_shots = len(partial_idx)

    coherence_score = 0
    for first_part, second_part in itertools.product(partial_idx, partial_idx):
        if first_part == second_part:
            continue
        if "maxmean" in args.search_objective:
            pairscore = pair_idx_to_score[first_part + "_" + second_part]["mean"]                                    
        elif "maxsum" in args.search_objective:
            pairscore = pair_idx_to_score[first_part + "_" + second_part]["sum"]                                    
        else:
            raise RuntimeError()
        coherence_score += pairscore
    coherence_score = coherence_score / num_shots / (num_shots - 1)

    return coherence_score

class StrategySearcherBase(ABC):
    def __init__(self, strategy_name):
        self.strategy_name = strategy_name

    @classmethod
    def from_strategy_name(cls, strategy_name):
        if strategy_name == "random":
            return RandomStrategySearhcer(strategy_name)
        elif strategy_name == "coherence":
            return ConherenceStrategySearcher(strategy_name)
        elif strategy_name == "avgsilver":
            return AverageSilverAccSearcher(strategy_name)
        else:
            raise RuntimeError("Not Implemented Yet")

    def get_combo_candidates(self, args, candidates, pair_idx_to_score, calib_idx_to_score, num_required):
        return NotImplementedError("Strategy not implemetned " + self.__class__.__name__)

    def get_num_of_combinations(self, candidates):
        return reduce((lambda x, y: x * y), [len(p) for p in candidates])


class RandomStrategySearhcer(StrategySearcherBase):
    def get_combo_candidates(self, args, candidates, pair_idx_to_score, calib_idx_to_score, num_required):
        random.seed(args.randseed)

        top_candidates = []
        num_candidates_each_shot = [len(p) for p in candidates]
        while True:
            selected = (random.randrange(l) for l in num_candidates_each_shot)
            if selected in top_candidates:
                continue
            top_candidates.append(selected)
            if len(top_candidates) == num_required:
                break

        top_candidates = [list(x) for x in top_candidates]
        return top_candidates


class ConherenceStrategySearcher(StrategySearcherBase):
    def get_combo_candidates(self, args, candidates, pair_idx_to_score, calib_idx_to_score, num_required):
            # select prompts by hard searching
        choice_generator = itertools.product(*[list(range(len(p))) for p in candidates])
        num_total_choice = reduce((lambda x, y: x * y), [len(p) for p in candidates])

        priority_q = BufferedHeap()


        for choice in tqdm(choice_generator, total=num_total_choice, desc="Searching"):
            partial_idx = ["p{}-c{}".format(*x) for x in enumerate(choice)]
            score = 0
            for first_part, second_part in itertools.product(partial_idx, partial_idx):
                if first_part == second_part:
                    continue
                if "maxmean" in args.search_objective:
                    pairscore = pair_idx_to_score[first_part + "_" + second_part]["mean"]
                elif "maxsum" in args.search_objective:
                    pairscore = pair_idx_to_score[first_part + "_" + second_part]["sum"]
                else:
                    raise RuntimeError()
                score += pairscore
            priority_q.push(choice, score)

        top_candidates = priority_q.get_topk(num_required)
        top_candidates = [list(x[1]) for x in top_candidates]

        return top_candidates


class BaseSilverAccuracySearcher(StrategySearcherBase):
    def get_indivisual_candidate_scores(self, calib_idx_to_greedy_eval, silver_eval_results):
        teacher_voted_preds = silver_eval_results["all_voted_predictions"]
        teacher_pred_confidence = [
            sum([voted_p == x for x in preds]) * 1.0 / len(preds)
            for (voted_p, preds) in zip(teacher_voted_preds, silver_eval_results["all_raw_predictions"])
        ]

        indivisual_candidate_scores = {}
        for k, hyp_eval_results in calib_idx_to_greedy_eval.items():
            student_voted_preds = hyp_eval_results["all_voted_predictions"]

            aggrement = [a == b for (a,b) in zip(student_voted_preds, teacher_voted_preds)]
            aggrement = np.array(aggrement, dtype=np.float32)

            if self.strategy_name in ["wghsilver"]:
                aggrement = aggrement * np.asarray(teacher_pred_confidence)
            elif self.strategy_name in ["avgsilver"]:
                pass
            else:
                raise RuntimeError("Not implemented")
            aggrement = np.mean(aggrement)
            indivisual_candidate_scores[k] = aggrement
        return indivisual_candidate_scores

    def get_combo_candidates(self, args, candidates, calib_idx_to_greedy_eval, silver_eval_results,
                                pair_idx_to_score, calib_idx_to_score, num_required):


        indivisual_candidate_scores = self.get_indivisual_candidate_scores(calib_idx_to_greedy_eval, silver_eval_results)

        cutoff = 4
        ind_fn = lambda x, y: f'p{x}-c{y}'
        while True:
            num_retrained = reduce((lambda x, y: x * y), [min(cutoff, len(p)) for p in candidates])
            if num_retrained > num_required or cutoff == max([len(p) for p in candidates]):
                break
            cutoff += 1

        scores_list_by_shots = [
            [indivisual_candidate_scores[ind_fn(p_idx, c_idx)] for c_idx in range(len(shots))]
                for (p_idx, shots) in enumerate(candidates)
        ]

        score_idx_bundles = [list(enumerate(scores_for_shots)) for scores_for_shots in scores_list_by_shots]
        score_idx_bundles = [sorted(b, key=lambda x: x[1], reverse=True) for b in score_idx_bundles]
        score_idx_bundles = [b[:cutoff] for b in score_idx_bundles]

        choice_generator = itertools.product(*score_idx_bundles)
        num_total_choice = reduce((lambda x, y: x * y), [len(b) for b in score_idx_bundles])

        priority_q = BufferedHeap()

        for bundle_choice in tqdm(choice_generator, total=num_total_choice, desc="Searching"):
            # print(bundle_choice)

            choice = [x[0] for x in bundle_choice]
            score = sum([x[1] for x in bundle_choice])

            priority_q.push(choice, score)

        top_candidates = priority_q.get_topk(num_required)
        top_candidates = [list(x[1]) for x in top_candidates]

        return top_candidates

class WeightedSilverAccSearcher(BaseSilverAccuracySearcher):
    def __init__(self, strategy_name):
        super().__init__(strategy_name)
        assert strategy_name == "wghsilver"

class AverageSilverAccSearcher(BaseSilverAccuracySearcher):
    def __init__(self, strategy_name):
        super().__init__(strategy_name)
        assert strategy_name == "avgsilver"
