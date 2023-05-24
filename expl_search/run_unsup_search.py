# search with validation
import sys
sys.path.append('.')

import os
import argparse
import random
import numpy as np

from tqdm import tqdm
from functools import reduce, partial

from utils import *
from api_utils import (
    run_completion_tasks_with_cache,
    config_args_and_api,
    register_base_args,
    DEFAULT_TRAIN_SEP,
    DEFAULT_COMPLETION_LEADING,
)

from run_manual import read_manual_prompt
from run_selective import (
    load_train_test_set,
    TaskHelper,
    run_evaluation
)

from task_evaluator import get_task_evaluator, print_tabular_results
from expl_search.run_cons_search import (
    register_cons_args,
    get_random_shots,
    leave_one_augment,
    evaluate_overgen_responses,
    score_of_pair_shot,
    leave_one_result_filename_func,
    leave_one_candidate_score_filename_func,
    query_with_predefined_shots,
)   


from expl_search.run_val_search import (
    get_tune_samples,
    get_valid_candidates,
    precache_set_candidates_by_strategy,
    get_greedy_acc_on_tune_set,
    val_search_combination_idx_validation_filename_func,
    val_search_combination_idx_final_test_filename_func,
    get_pair_scores_and_calib_scores,
    val_search_top_combinations_filename_func
)   

from expl_search.strategy_searcher import StrategySearcherBase
from expl_search.strategy_explorer import StrategyExplorerBase
from expl_search.inspections import inspection_print
# 
def register_unsup_args(parser):
    parser.add_argument('--tune_split', type=str, default="train", choices=["train", "test"])
    # how to select the tune samples
    parser.add_argument('--tune_selection', type=str, default="adaptive", choices=["adaptive", "randtail", "firstk"])
    parser.add_argument('--num_tune', type=int, default=256)

    parser.add_argument("--teacher_times_search", type=int, default=32)
    parser.add_argument('--teacher_search_strategy', default="random", choices=["random", "coherence"])
    parser.add_argument('--teacher_batch_size', type=int, default=None)
    parser.add_argument("--teacher_num_samples", type=int, default=None)
    parser.add_argument("--teacher_temperature", type=float, default=None)
    parser.add_argument("--teacher_explore_strategy", default="dummy")

    parser.add_argument("--student_times_search", type=int, default=32)
    # avgsilver: OsACC, coherence: OsLL
    parser.add_argument('--student_search_strategy', default="random", choices=[
        "random", "coherence", "avgsilver"])

    parser.add_argument('--student_batch_size', type=int, default=None)
    parser.add_argument("--student_num_samples", type=int, default=None)
    parser.add_argument("--student_temperature", type=float, default=None)
    parser.add_argument("--student_selection", type=str, default="average", choices=["weighted", "average"])
    parser.add_argument("--student_explore_strategy", default="dummy")

    parser.add_argument('--search_strategy', default=None,)
    parser.add_argument('--tune_batch_size', type=int, default=None)
    parser.add_argument("--tune_num_samples", type=int, default=None)
    parser.add_argument("--tune_temperature", type=float, default=None)


    parser.add_argument('--combo_buffer_size', default=2048)
    parser.add_argument('--num_final_eval_samples', type=int, default=-1)

    parser.add_argument('--do_score_only', default=False, action="store_true")

# TODO: quick implementation for now
def apply_selected_idxes(valid_shots, selected_idxes):
    if isinstance(selected_idxes[0], str):
        selected_shots = []
        for idx in selected_idxes:
            assert "-" in idx
            pid, cid = idx.split("-")
            pid = int(pid[1:])
            cid = int(cid[1:])
            selected_shots.append(valid_shots[pid][cid])
    elif isinstance(selected_idxes[0], int):
        selected_shots = [p[c_idx] for p, c_idx in zip(valid_shots, selected_idxes)]
    else:
        raise RuntimeError("invalid combonation idxes type")
    return selected_shots


def run_preds_with_tune_set(args, valid_shots, top_candidates, task_specific_helper, tune_data):
    all_eval_results = []
    all_responses = []

    combination_explorer = StrategyExplorerBase.from_strategy_name(args.teacher_explore_strategy, top_candidates)
    combination_explorer.init_status(args)

    search_step = 0
    explored_combos = []
    while search_step < args.teacher_times_search:
        # get next thing to explore
        set_idx, selected_idxes = combination_explorer.get_next_combo(explored_combos)

        selected_shots = apply_selected_idxes(valid_shots, selected_idxes)

        idx_result_filename_func = partial(val_search_combination_idx_validation_filename_func, set_idx)
        eval_results, idx_responses = query_with_predefined_shots(args, task_specific_helper, selected_shots, tune_data,
            idx_result_filename_func,
            query_args={
                "engine": args.test_engine,
                "num_samples": args.tune_num_samples,
                "temperature": args.tune_temperature,
                "batch_size": args.tune_batch_size
                },
            return_responses=True
            )
        # print(set_idx, eval_results)
        all_eval_results.append(eval_results)
        all_responses.append(idx_responses)

        explored_combos.append({"set_index": set_idx})
        search_step += 1

    return all_eval_results, all_responses

def score_student_candidate_hypothesis(args, hyp_eval_results, teacher_voted_preds, teacher_pred_confidence):
    student_voted_preds = hyp_eval_results["all_voted_predictions"]
    if args.student_selection == "weighted":
        aggrement = [a == b for (a,b) in zip(student_voted_preds, teacher_voted_preds)]
        aggrement = np.array(aggrement, dtype=np.float32)
        aggrement = aggrement * np.asarray(teacher_pred_confidence)
        return np.mean(aggrement)
    elif args.student_selection == "average":
        aggrement = [a == b for (a,b) in zip(student_voted_preds, teacher_voted_preds)]
        aggrement = np.array(aggrement, dtype=np.float32)
        return np.mean(aggrement)
    else:
        raise RuntimeError("Unsupported student selection")

def mbr_search_with_val_set(args, valid_shots, top_candidates, task_specific_helper, tune_data, teacher_merged_eval_results):

    teacher_voted_preds = teacher_merged_eval_results["all_voted_predictions"]
    teacher_pred_confidence = [
        sum([voted_p == x for x in preds]) * 1.0 / len(preds)
        for (voted_p, preds) in zip(teacher_voted_preds, teacher_merged_eval_results["all_raw_predictions"])
    ]
    # assert len(set([len(preds) for preds in teacher_merged_eval_results["all_raw_predictions"]])) == 1

    best_set = None
    best_score = float("-inf")

    combination_explorer = StrategyExplorerBase.from_strategy_name(args.student_explore_strategy, top_candidates)
    combination_explorer.init_status(args)

    search_step = 0
    explored_combos = []
    while search_step < args.student_times_search:
        # get next thing to explore
        set_idx, selected_idxes = combination_explorer.get_next_combo(explored_combos)

        selected_shots = apply_selected_idxes(valid_shots, selected_idxes)
        idx_result_filename_func = partial(val_search_combination_idx_validation_filename_func, set_idx)
        hyp_eval_results, idx_responses = query_with_predefined_shots(args, task_specific_helper, selected_shots, tune_data,
            idx_result_filename_func,
            query_args={
                "engine": args.test_engine,
                "num_samples": args.tune_num_samples,
                "temperature": args.tune_temperature,
                "batch_size": args.tune_batch_size
                },
                return_verbose=True,
                return_responses=True,
            )

        score_of_hyp = score_student_candidate_hypothesis(args, hyp_eval_results, teacher_voted_preds, teacher_pred_confidence)
        print(set_idx, score_of_hyp)
        if score_of_hyp > best_score:
            best_score = score_of_hyp
            best_set = (set_idx, selected_shots, hyp_eval_results)

        exploration = {"set_index": set_idx}

        explored_combos.append(exploration)
        search_step += 1

    inspection_print("BestSet", best_set[0], best_score, best_set[2]["accuracy"], best_set[2]["avg_normlogprob"])
    return best_set, explored_combos

def eval_ensemble_on_tune_set(args, tune_data, all_tune_responses, all_tune_eval_results):
    # single split performance
    acc_of_each_prompt = [x["accuracy"] * 100 for x in all_tune_eval_results]
    print("NUM {:.2f} MAX ACC {:.2f} AVG ACC {:.2f}".format(len(acc_of_each_prompt),
        max(acc_of_each_prompt), sum(acc_of_each_prompt) / len(acc_of_each_prompt)))

    merged = [list(itertools.chain(*[r[i] for r in all_tune_responses])) for i in range(len(tune_data))]
    eval_results = run_evaluation(args, tune_data, merged, return_verbose=True)
    return eval_results

def unsup_search_seed_validation_filename_func(args):
    resp_file = leave_one_result_filename_func(args)

    if args.tune_split == "train" and args.do_manual_prompt:
        split_identifier = "tr" + str(args.slice_train) + "-" + str(args.slice_train + args.num_train)
    elif args.tune_split == "test":
        split_identifier = "tt" + str(args.slice_dev) + "-" + str(args.slice_dev + args.num_dev)
    else:
        split_identifier = args.tune_split

    resp_file = resp_file.replace("--predictions.json",
        "--vidseed--vst{}--vse{}--vt{}--vns{}--vt{}--vval.json".format(
            split_identifier, args.tune_selection, args.num_tune, args.tune_num_samples, args.tune_temperature,
        ))

    resp_file = resp_file.replace("--", "=")

    return resp_file

def search_with_validation(args):
    task_specific_helper = TaskHelper.from_taskname(args.task, args.style_template)

    train_data, test_data = load_train_test_set(args)
    tune_data = get_tune_samples(args, train_data, test_data)
    valid_shots = get_valid_candidates(args, task_specific_helper, train_data, test_data)
    full_candidates_for_pairscores = [p[:] for p in valid_shots]

    num_total_choice = reduce((lambda x, y: x * y), [len(p) for p in valid_shots])

    print("Num total", reduce((lambda x, y: x * y), [len(p) for p in valid_shots]))

    # teacher loop
    args.search_strategy = args.teacher_search_strategy
    args.tune_num_samples = args.teacher_num_samples
    args.tune_temperature = args.teacher_temperature
    args.tune_batch_size = args.teacher_batch_size
    teacher_candidates = precache_set_candidates_by_strategy(args, valid_shots, full_candidates_for_pairscores,
        task_specific_helper, tune_data)
    teacher_ind_eval_results, teacher_eval_responses = run_preds_with_tune_set(
        args, valid_shots, teacher_candidates, task_specific_helper, tune_data)
    teacher_merged_eval_results = eval_ensemble_on_tune_set(args, tune_data, teacher_eval_responses, teacher_ind_eval_results)
    print_tabular_results("TUNETEA" + str(args.teacher_times_search), teacher_merged_eval_results)

    # student loop
    args.search_strategy = args.student_search_strategy
    args.tune_num_samples = args.student_num_samples
    args.tune_temperature = args.student_temperature
    args.tune_batch_size = args.student_batch_size
    student_candidates = precache_set_candidates_by_strategy(args, valid_shots, full_candidates_for_pairscores,
        task_specific_helper, tune_data, silver_eval_results=teacher_merged_eval_results)
    (selected_idx, tune_selected_shots, tune_eval_results), explored_combos = mbr_search_with_val_set(
        args, valid_shots, student_candidates, task_specific_helper, tune_data, teacher_merged_eval_results)
    print_tabular_results("TUNESTU", tune_eval_results)

    if args.do_score_only:
        return

    final_result_filename_func = partial(val_search_combination_idx_final_test_filename_func, selected_idx)

    args.num_eval_samples = args.num_final_eval_samples
    final_eval_results = query_with_predefined_shots(args, task_specific_helper, tune_selected_shots,
                        test_data, final_result_filename_func)
    final_eval_num_samples = args.num_final_eval_samples if args.num_final_eval_samples > 0 else args.test_num_samples
    print_tabular_results("GREEDY" if args.test_num_samples == 1 else f"VOTE{final_eval_num_samples}", final_eval_results)

def main():
    parser = argparse.ArgumentParser()
    register_base_args(parser)
    register_cons_args(parser)
    register_unsup_args(parser)
    args = parser.parse_args()

    # post process args
    assert args.task is not None
    assert args.search_strategy is None
    assert args.tune_num_samples is None
    assert args.tune_temperature is None

    tune_selection = args.tune_selection
    if tune_selection == "adaptive":
        tune_selection = "randtail" if args.tune_split == "train" else "firstk"
    args.tune_selection = tune_selection

    if args.aug_engine is None:
        args.aug_engine = args.engine
    if args.test_engine is None:
        args.test_engine = args.engine

    config_args_and_api(args)
    search_with_validation(args)


if __name__=='__main__':
    main()
