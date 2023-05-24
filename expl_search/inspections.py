import sys
sys.path.append('.')

import numpy as np
import itertools
import math
import random
from tqdm import tqdm


from collections import OrderedDict

from utils import *
from api_utils import (
    DEFAULT_TRAIN_SEP,
    DEFAULT_COMPLETION_LEADING,
)

def inspection_print(*args, **kwargs):
    print("INSPECT", *args, **kwargs)

# TODO: quick implementation for now
# def apply_selected_idxes(valid_shots, selected_idxes):
#     if isinstance(selected_idxes[0], str):
#         selected_shots = []
#         for idx in selected_idxes:
#             assert "-" in idx
#             pid, cid = idx.split("-")
#             pid = int(pid[1:])
#             cid = int(cid[1:])
#             selected_shots.append(valid_shots[pid][cid])
#     elif isinstance(selected_idxes[0], int):
#         selected_shots = [p[c_idx] for p, c_idx in zip(valid_shots, selected_idxes)]
#     else:
#         raise RuntimeError("invalid combonation idxes type")
#     return selected_shots

def sort_candidate_choices_per_problem(candidates):
    num_shots = len(candidates[0])
    is_str_idx = isinstance(candidates[0][0], str)

    choices = [[] for _ in range(num_shots)]
    for combo in candidates:
        if is_str_idx:
            for idx in combo:
                pid, cid = idx.split("-")
                pid = int(pid[1:])
                cid = int(cid[1:])
                choices[pid].append(cid)
        else:
            for pid, cid in enumerate(combo):
                choices[pid].append(cid)

    sorted_choices = OrderedDict()
    for i, c in enumerate(choices):                
        sorted_idx = sorted(list(set(c)))
        sorted_choices[f'p{i}'] = ", ".join(map(str,sorted_idx))

    return sorted_choices

def inspect_candidate_choices(args, top_candidates, explored_combos, best_idx):
    if not args.do_inspect:
        return

    # choice info
    explored_candidates = [top_candidates[c["set_index"]] for c in explored_combos]

    explored_stats = sort_candidate_choices_per_problem(explored_candidates)
    inspection_print("EXPLORED COVERAGE")
    for k, v in explored_stats.items():
        inspection_print(k, v)

    top_stats = sort_candidate_choices_per_problem(top_candidates)
    inspection_print("TOP COVERAGE")
    for k, v in top_stats.items():
        inspection_print(k, v)

    inspection_print("SELECTED", best_idx, top_candidates[best_idx])

def _get_indivisual_candidate_scores(strategy_name, calib_idx_to_greedy_eval, silver_eval_results):
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

        if strategy_name in ["wghsilver", "cohwghsilver", "cohwghlog"]:
            aggrement = aggrement * np.asarray(teacher_pred_confidence)
        elif strategy_name in ["avgsilver", "cohavgsilver", "cohavglog"]:
            pass
        else:
            raise RuntimeError("Not implemented")
        aggrement = np.mean(aggrement)
        indivisual_candidate_scores[k] = aggrement
    return indivisual_candidate_scores

def _calc_coherence_score_of_a_combination(args, partial_idx, pair_idx_to_score):
    num_shots = len(partial_idx)

    coherence_score = 0
    for first_part, second_part in itertools.product(partial_idx, partial_idx):
        if first_part == second_part:
            continue
        if "order" in args.search_objective:
            first_shot_idx = int(first_part.split("-")[0][1:])
            second_shot_idx = int(second_part.split("-")[0][1:])
            if first_shot_idx > second_shot_idx:
                continue
        if "maxmean" in args.search_objective:
            pairscore = pair_idx_to_score[first_part + "_" + second_part]["mean"]                                    
        elif "maxsum" in args.search_objective:
            pairscore = pair_idx_to_score[first_part + "_" + second_part]["sum"]                                    
        else:
            raise RuntimeError()
        coherence_score += pairscore
    coherence_score = coherence_score / num_shots / (num_shots - 1)
    if 'order' in args.search_objective:
        coherence_score = coherence_score * 2

    return coherence_score

def _inspect_stats(header, x):
    x = np.asarray(x)
    inspection_print(header, "AVG {:.3f} STD {:.3f} MIN {:.3f} MAX {:.3f}".format(
        np.mean(x), np.std(x), np.min(x), np.max(x)))

def _calc_silver_score(strategy_name, partial_idx, indivisual_candidate_scores, num_shots):
    EPOSILON=0.01
    if strategy_name in ["cohwghsilver", "cohavgsilver"]:
        silver_score = sum([indivisual_candidate_scores[first_part] for first_part in partial_idx]) / num_shots
        silver_score = math.log(silver_score)
    elif strategy_name in ["cohwghlog", "cohavglog"]:
        silver_score = sum([math.log(max(indivisual_candidate_scores[first_part], EPOSILON)) for first_part in partial_idx]) / num_shots
    else:
        raise RuntimeError("Not implemented")
    return silver_score

def inspect_student_candidate_info(args, combinations, candidates, calib_idx_to_greedy_eval, silver_eval_results, pair_idx_to_score, calib_idx_to_score):
    indivisual_candidate_scores = _get_indivisual_candidate_scores("avgsilver", calib_idx_to_greedy_eval, silver_eval_results)
    num_shots = len(candidates)

    if "calib" in args.search_objective:
        for pair_idx in pair_idx_to_score:
            second_part = pair_idx.split("_")[1]
            base_score = calib_idx_to_score[second_part]
            orig_score = pair_idx_to_score[pair_idx]
            pair_idx_to_score[pair_idx] = {
                "sum": orig_score["sum"] - base_score["sum"],
                "mean": orig_score["mean"] - base_score["mean"],
            }

    
    top_avglog_scores = []
    top_avgsilver_scores = []
    top_coh_scores = []
    for choice in combinations:
        partial_idx = ["p{}-c{}".format(*x) for x in enumerate(choice)]
        coherence_score = _calc_coherence_score_of_a_combination(args, partial_idx, pair_idx_to_score)
        avglog_score = _calc_silver_score("cohavglog", partial_idx, indivisual_candidate_scores, num_shots)
        avgsilver_score = _calc_silver_score("cohavgsilver", partial_idx, indivisual_candidate_scores, num_shots)
        top_avglog_scores.append(avglog_score)
        top_avgsilver_scores.append(avgsilver_score)
        top_coh_scores.append(coherence_score)
    top_avglog_scores = np.asarray(top_avglog_scores)
    top_avgsilver_scores = np.asarray(top_avgsilver_scores)
    top_coh_scores = np.asarray(top_coh_scores)
    # TOP 32 stats
    _inspect_stats("AvgLog32", top_avglog_scores[:32])
    _inspect_stats("AvgSilver32", top_avgsilver_scores[:32])
    _inspect_stats("COHERENCE32", top_coh_scores[:32])
    # TOP all stats
    _inspect_stats("AvgLog", top_avglog_scores)
    _inspect_stats("AvgSilver", top_avgsilver_scores)
    _inspect_stats("COHERENCE", top_coh_scores)

    top_avglog_scores = []
    top_avgsilver_scores = []
    top_coh_scores = []
    num_candidates_each_shot = [len(p) for p in candidates]
    for _ in tqdm(range(1000000)):
        choice = (random.randrange(l) for l in num_candidates_each_shot)
        partial_idx = ["p{}-c{}".format(*x) for x in enumerate(choice)]
        coherence_score = _calc_coherence_score_of_a_combination(args, partial_idx, pair_idx_to_score)
        avglog_score = _calc_silver_score("cohavglog", partial_idx, indivisual_candidate_scores, num_shots)
        avgsilver_score = _calc_silver_score("cohavgsilver", partial_idx, indivisual_candidate_scores, num_shots)
        top_avglog_scores.append(avglog_score)
        top_avgsilver_scores.append(avgsilver_score)
        top_coh_scores.append(coherence_score)
    top_avglog_scores = np.asarray(top_avglog_scores)
    top_avgsilver_scores = np.asarray(top_avgsilver_scores)
    top_coh_scores = np.asarray(top_coh_scores)

    # TOP all stats
    _inspect_stats("SampleAvgLog", top_avglog_scores)
    _inspect_stats("SampleAvgSilver", top_avgsilver_scores)
    _inspect_stats("SampleCOHERENCE", top_coh_scores)


def augment_plot_info(args, explored_combos, combinations, candidates, calib_idx_to_greedy_eval, silver_eval_results, pair_idx_to_score, calib_idx_to_score):
    indivisual_candidate_scores = _get_indivisual_candidate_scores("avgsilver", calib_idx_to_greedy_eval, silver_eval_results)
    num_shots = len(candidates)

    assert "calib" not in args.search_objective
    # if "calib" in args.search_objective:
    #     for pair_idx in pair_idx_to_score:
    #         second_part = pair_idx.split("_")[1]
    #         base_score = calib_idx_to_score[second_part]
    #         orig_score = pair_idx_to_score[pair_idx]
    #         pair_idx_to_score[pair_idx] = {
    #             "sum": orig_score["sum"] - base_score["sum"],
    #             "mean": orig_score["mean"] - base_score["mean"],
    #         }

    for exploration in explored_combos:
        choice = combinations[exploration["set_index"]]
        partial_idx = ["p{}-c{}".format(*x) for x in enumerate(choice)]
        coherence_score = _calc_coherence_score_of_a_combination(args, partial_idx, pair_idx_to_score)
        avglog_score = _calc_silver_score("cohavglog", partial_idx, indivisual_candidate_scores, num_shots)
        avgsilver_score = _calc_silver_score("cohavgsilver", partial_idx, indivisual_candidate_scores, num_shots)
        exploration["coherence_score"] = coherence_score
        exploration["avglog_score"] = avglog_score
        exploration["avgacc_score"] = avgsilver_score

        individual_answer_avg_logprob = [calib_idx_to_score[p]["mean"] for p in partial_idx]
        # print(individual_answer_avg_logprob)
        individual_answer_avg_logprob = sum(individual_answer_avg_logprob) / num_shots
        exploration["individual_answer_avg_logprob"] = individual_answer_avg_logprob


def safe_offset_index_indexer(token_offset, offset):
    # if offset in token_offset:
    #     tok_idx = token_offset.index(offset)
    # elif offset >= token_offset[-1]:
    #     tok_idx = -1
    # else: 
    #     answer_start_tok_idx = next(filter(lambda x: token_offset[x] >= answer_offset, range(len(token_offset))))
    if offset > token_offset[-1]:
        return -1
    else:
        return next(filter(lambda x: token_offset[x] >= offset, range(len(token_offset))))

def score_of_range(token_offset, token_logprobs, tokens, start_offset, end_offset, including_ending=1):
    # completion_start_tok_idx = token_offset.index(start_offset)
    # completion_end_tok_idx = token_offset.index(end_offset)

    completion_start_tok_idx = safe_offset_index_indexer(token_offset, start_offset)
    completion_end_tok_idx = safe_offset_index_indexer(token_offset, end_offset)
    # return len(tokens) - completion_start_tok_idx

    if completion_start_tok_idx >= (completion_end_tok_idx + including_ending):
        return {"sum": 0., "mean":0. , "num": 0}
    tok_scores = token_logprobs[completion_start_tok_idx:completion_end_tok_idx + including_ending]
    toks = tokens[completion_start_tok_idx:completion_end_tok_idx + including_ending]
    # print("".join(toks))
    if tok_scores[0] is None:
        tok_scores[0] = 0
    tok_scores = np.array(tok_scores)
    
    return {"sum": tok_scores.sum(), "mean": tok_scores.mean(), "num": len(toks)}

def augment_prompt_score_criteron(exploration, response):
    tokens = response["logprobs"]["tokens"]
    token_offset = response["logprobs"]["text_offset"]
    token_logprobs = response["logprobs"]["token_logprobs"]
    tokens = response["logprobs"]["tokens"]

    prompt = response["prompt"]

    # score full prompts
    full_prompt_start_offset = 0
    full_prompt_end_offset = response["prompt"].rfind(DEFAULT_TRAIN_SEP)

    prompt_full_stats = score_of_range(token_offset, token_logprobs, tokens, full_prompt_start_offset, full_prompt_end_offset)

    segment_start = 0

    acc_answer_len = 0
    acc_answer_logprob = 0
    while True:
        segment_end = prompt.find(DEFAULT_TRAIN_SEP, segment_start)
        if segment_end == -1:
            break
        answer_start = prompt.find(DEFAULT_COMPLETION_LEADING, segment_start) + len(DEFAULT_COMPLETION_LEADING)
        segment_start = segment_end + len(DEFAULT_TRAIN_SEP)
        single_answer_stats = score_of_range(token_offset, token_logprobs, tokens, answer_start, segment_end, including_ending=0)
        acc_answer_len += single_answer_stats["num"]
        acc_answer_logprob += single_answer_stats["sum"]
    answer_avg_logprob = acc_answer_logprob / acc_answer_len
    exploration["prompt_full_avg_logprob"] = prompt_full_stats["mean"]
    exploration["prompt_answer_avg_logprob"] = answer_avg_logprob

if __name__=="__main__":
    exploreation = {}
    resps = read_json("expl_search/misc/loaug=gsm=tr3072-4096=rand0=numshot8=stydefault=engcode-davinci-002=numsamp40=temp0.7=shstgrandcoh4=sd0=objmaxmean=top2048=vid44=vsttrain=vserandtail=vt256=vns1=vt0.7=vval.json")
    augment_prompt_score_criteron(exploreation, resps[0][0][0])
    print(exploreation)
