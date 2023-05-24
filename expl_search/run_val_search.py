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
    leave_one_candidate_pool_filename_funct,
    leave_one_candidate_score_filename_func,
    query_with_predefined_shots,
    read_and_prepare_candidates,
)   

from expl_search.strategy_searcher import StrategySearcherBase

def register_validation_args(parser):
    parser.add_argument('--tune_split', type=str, default="train", choices=["train", "test"])
    # how to select the tune samples
    parser.add_argument('--tune_selection', type=str, default="adaptive", choices=["adaptive", "randtail", "firstk"])
    parser.add_argument('--num_tune', type=int, default=256)

    parser.add_argument('--tune_batch_size', type=int, default=1)
    parser.add_argument("--tune_num_samples", type=int, default=1)
    parser.add_argument("--tune_temperature", type=float, default=0.0)

    parser.add_argument("--times_search", type=int, default=128)

    parser.add_argument('--search_strategy', required=None, choices=["random", "coherence", "avgsliver"])
    parser.add_argument('--combo_buffer_size', default=2048)


def get_random_tail_samples(rand_seed, data_split, num_samples):
    np.random.seed(rand_seed)
    # make it consistent with what's in run_selective
    randome_scores = np.random.rand(len(data_split))

    selected_idx = sorted(enumerate(randome_scores), key=lambda x: x[1], reverse=True)[-num_samples:]
    selected_idx = [x[0] for x in selected_idx]
    selected_shots = [data_split[i] for i in selected_idx]

    return selected_shots

def get_valid_candidates(args, task_specific_helper, train_data, test_data):
    # get shots
    if args.do_manual_prompt:
        manual_prompt = read_manual_prompt(args.task, args.manual_prompt_id, args.style_template)
        shots = manual_prompt.split(DEFAULT_TRAIN_SEP)
        args.num_shots = len(shots)
    else:
        raw_shots = get_random_shots(args.randseed, train_data, args.num_shots)
        encoded_prompt = task_specific_helper.prompt_func(test_data[0], raw_shots)
        shots = encoded_prompt.split(DEFAULT_TRAIN_SEP)[:-1]

    overgen_responses = leave_one_augment(args, shots, task_specific_helper)
    if args.do_inspect:
        evaluate_overgen_responses(args, shots, overgen_responses)
    
    valid_shots = read_and_prepare_candidates(args, shots, overgen_responses)
    # if args.do_dryrun or args.do_inspect:
    print("Valid search candidates", [len(x) for x in valid_shots])

    return valid_shots

def get_tune_samples(args, train_data, test_data):
    train_data, test_data = load_train_test_set(args)

    data_split = train_data if args.tune_split == "train" else test_data
    tune_selection = args.tune_selection

    if tune_selection == "randtail":
        tune_set = get_random_tail_samples(args.randseed, data_split, args.num_tune)
    elif tune_selection == "firstk":
        tune_set = data_split[:args.num_tune]
    else:
        raise RuntimeError("Invalid branch")

    return tune_set

def val_search_top_combinations_filename_func(args):
    # all the shot related stuff
    resp_file = leave_one_candidate_pool_filename_funct(args)

    if args.search_strategy in ["coherence"]:
        resp_file = resp_file.replace("--predictions.json", "--searchstg{}--obj{}--top{}--combinations.json".format(
                args.search_strategy, args.search_objective, args.combo_buffer_size))
    elif args.search_strategy == "random":
        resp_file = resp_file.replace("--predictions.json", "--searchstg{}--searchseed{}--top{}--combinations.json".format(
            args.search_strategy, args.randseed, args.combo_buffer_size))
    # bug here: related to teacherstratego not listed
    elif args.search_strategy in ["avgsilver"]:
        if args.tune_split == "train" and args.do_manual_prompt:
            split_identifier = "tr" + str(args.slice_train) + "-" + str(args.slice_train + args.num_train)
        elif args.tune_split == "test":
            split_identifier = "tt" + str(args.slice_dev) + "-" + str(args.slice_dev + args.num_dev)
        else:
            split_identifier = args.tune_split
        additional_info_identifier = [split_identifier, args.tune_selection, args.num_tune,
            "" if  args.teacher_search_strategy == "random" else (args.search_objective
                + args.teacher_search_strategy),
            args.teacher_times_search, args.teacher_num_samples, args.teacher_temperature,
            args.student_num_samples, args.student_temperature]
        additional_info_identifier = "".join(map(str, additional_info_identifier))
        resp_file = resp_file.replace("--predictions.json", "--shstg{}--{}--top{}--combinations.json".format(
            args.search_strategy, additional_info_identifier, args.combo_buffer_size))
    else:
        raise RuntimeError("Search strategy")

    return resp_file

def val_search_combination_idx_validation_filename_func(idx, args):
    resp_file = val_search_top_combinations_filename_func(args)


    if args.tune_split == "train" and args.do_manual_prompt:
        split_identifier = "tr" + str(args.slice_train) + "-" + str(args.slice_train + args.num_train)
    elif args.tune_split == "test":
        split_identifier = "tt" + str(args.slice_dev) + "-" + str(args.slice_dev + args.num_dev)
    else:
        split_identifier = args.tune_split

    resp_file = resp_file.replace("--combinations.json",
        "--vid{}--vst{}--vse{}--vt{}--vns{}--vt{}--vval.json".format(
            idx, split_identifier, args.tune_selection, args.num_tune, args.tune_num_samples, args.tune_temperature,
        ))

    resp_file = resp_file.replace("--", "=")

    return resp_file


def val_search_greedy_eval_idx_validation_filename_func(idx, args):
    assert args.search_strategy == "avgsilver"

    resp_file = leave_one_candidate_pool_filename_funct(args)
    resp_file = resp_file.replace("--predictions.json", "--searchstg{}--top{}--combinations.json".format(
            "greedy", args.combo_buffer_size))

    if args.tune_split == "train" and args.do_manual_prompt:
        split_identifier = "tr" + str(args.slice_train) + "-" + str(args.slice_train + args.num_train)
    elif args.tune_split == "test":
        split_identifier = "tt" + str(args.slice_dev) + "-" + str(args.slice_dev + args.num_dev)
    else:
        split_identifier = args.tune_split

    resp_file = resp_file.replace("--combinations.json",
        "--vid{}--vst{}--vse{}--vt{}--vns{}--vt{}--vval.json".format(
            idx, split_identifier, args.tune_selection, args.num_tune, args.tune_num_samples, args.tune_temperature,
        ))

    resp_file = resp_file.replace("--", "=")

    return resp_file

def val_search_combination_idx_final_test_filename_func(idx, args):
    resp_file = val_search_top_combinations_filename_func(args)

    resp_file = resp_file.replace("--combinations.json",
        "--vid{}--dv{}-{}--feng{}--fns{}--vt{}--ftest.json".format(
            idx, 
            args.slice_dev, args.slice_dev + args.num_dev,
            args.test_engine, args.test_num_samples, args.test_temperature,
        ))
    resp_file = resp_file.replace("--", "=")

    return resp_file

def get_pair_scores_and_calib_scores(args, candidates):
    #p1-c0-p2-c0
    pairs = []
    for p1_idx, p1 in enumerate(candidates):
        for p2_idx, p2 in enumerate(candidates):
            if p1_idx == p2_idx:
                continue
            for c1_idx, c1 in enumerate(p1):
                for c2_idx, c2 in enumerate(p2):
                    pair_idx = f"p{p1_idx}-c{c1_idx}_p{p2_idx}-c{c2_idx}"
                    prompt = "{}{}{}".format(c1, DEFAULT_TRAIN_SEP, c2)
                    pairs.append((pair_idx, prompt))

    score_cache_name = leave_one_candidate_score_filename_func(args)
    prompts_to_complete = [[x[1]] for x in pairs]

    if args.run_prediction:
        _num_samples, _temperature, _batch_size = args.num_samples, args.temperature, args.batch_size, 
        args.num_samples = 1
        args.temperature = 0.0
        args.batch_size = args.score_batch_size

        responses = run_completion_tasks_with_cache(args, score_cache_name, prompts_to_complete, 0)
        args.num_samples, args.temperature, args.batch_size,  = _num_samples, _temperature, _batch_size
    else:
        responses = read_json(score_cache_name)
    

    responses = flatten_nested_list(responses)
    responses = flatten_nested_list(responses)

    calib_idx_to_score = {}
    # for calibration
    for p1_idx, p1 in enumerate(candidates):
        for c1_idx, c1 in enumerate(p1):
            pair_prefix = f"p{p1_idx}-c{c1_idx}_"
            first_idx = next((i for i in range(len(pairs)) if pairs[i][0].startswith(pair_prefix)))
            base_resp = responses[first_idx]
            calib_idx_to_score[f"p{p1_idx}-c{c1_idx}"] = score_of_pair_shot(args, base_resp, score_base=True)

    pair_idx_to_score = {}
    for (pair_idx, prompt), resp in zip(pairs, responses):
        assert prompt == resp["prompt"]
        pair_idx_to_score[pair_idx] = score_of_pair_shot(args, resp)

    return pair_idx_to_score, calib_idx_to_score

def get_greedy_acc_on_tune_set(args, candidates, task_specific_helper, tune_data, return_full_eval_results=False):
    calib_idx_to_greedy_acc = {}
    for p1_idx, p1 in enumerate(candidates):
        for c1_idx, c1 in enumerate(p1):
            print("Querying", p1_idx, c1_idx)
            greedy_result_filename_func = partial(val_search_greedy_eval_idx_validation_filename_func, f"can{p1_idx}-{c1_idx}")
            eval_results = query_with_predefined_shots(args, task_specific_helper, [c1], tune_data,
                greedy_result_filename_func,
                query_args={
                    "engine": args.test_engine,
                    "num_samples": args.tune_num_samples,
                    "temperature": args.tune_temperature,
                    "batch_size": args.tune_batch_size
                    },
                return_verbose=return_full_eval_results
                )
            if return_full_eval_results:
                calib_idx_to_greedy_acc[f"p{p1_idx}-c{c1_idx}"] = eval_results
            else:
                calib_idx_to_greedy_acc[f"p{p1_idx}-c{c1_idx}"] = eval_results["accuracy"]
    return calib_idx_to_greedy_acc

def precache_set_candidates_by_strategy(args, candidates, full_candidates_for_pairscores, task_specific_helper, tune_data, silver_eval_results=None):
    args.stragey_searcher = StrategySearcherBase.from_strategy_name(args.search_strategy)
    pair_idx_to_score, calib_idx_to_score = get_pair_scores_and_calib_scores(args, full_candidates_for_pairscores)
    top_combinations_cache_name = val_search_top_combinations_filename_func(args)

    if os.path.exists(top_combinations_cache_name) and not args.force_override:
        print("Combination Cache", top_combinations_cache_name)
        top_candidates = read_json(top_combinations_cache_name)
        return top_candidates
    if args.search_strategy in ["random", "coherence"]:
        top_candidates = args.stragey_searcher.get_combo_candidates(args, candidates,
            pair_idx_to_score, calib_idx_to_score, args.combo_buffer_size)
    elif args.search_strategy in [ "avgsilver"]:
        assert silver_eval_results is not None
        calib_idx_to_greedy_eval = get_greedy_acc_on_tune_set(args, candidates, task_specific_helper, tune_data, return_full_eval_results=True)
        top_candidates = args.stragey_searcher.get_combo_candidates(args, candidates, calib_idx_to_greedy_eval, silver_eval_results,
            pair_idx_to_score, calib_idx_to_score, args.combo_buffer_size)
    else:
        raise NotImplementedError("Unsupported search strategy")

    dump_json(top_candidates, top_combinations_cache_name)
    return top_candidates

def cons_search_with_val_set(args, valid_shots, top_candidates, task_specific_helper, tune_data):
    best_set = None
    best_acc = .0

    for set_idx, selected_idxes in enumerate(top_candidates[:args.times_search]):
        selected_shots = [p[c_idx] for p, c_idx in zip(valid_shots, selected_idxes)]
        idx_result_filename_func = partial(val_search_combination_idx_validation_filename_func, set_idx)
        eval_results = query_with_predefined_shots(args, task_specific_helper, selected_shots, tune_data,
            idx_result_filename_func,
            query_args={
                "engine": args.test_engine,
                "num_samples": args.tune_num_samples,
                "temperature": args.tune_temperature,
                "batch_size": args.tune_batch_size
                }
            )
        print(eval_results)
        if eval_results["accuracy"] > best_acc:
            best_acc = eval_results["accuracy"] 
            best_set = (set_idx, selected_shots, eval_results)
    print(best_set[0], best_set[2])
    print_tabular_results("TUNE_ACC", best_set[2])
    # print(DEFAULT_TRAIN_SEP.join(best_set[1]))
    return best_set

def search_with_validation(args):
    task_specific_helper = TaskHelper.from_taskname(args.task, args.style_template)

    train_data, test_data = load_train_test_set(args)
    tune_data = get_tune_samples(args, train_data, test_data)
    valid_shots = get_valid_candidates(args, task_specific_helper, train_data, test_data)
    full_candidates_for_pairscores = [p[:] for p in valid_shots]


    top_candidates = precache_set_candidates_by_strategy(args, valid_shots, full_candidates_for_pairscores, task_specific_helper, tune_data)

    selected_idx, tune_selected_shots, tune_eval_results = cons_search_with_val_set(
        args, valid_shots, top_candidates, task_specific_helper, tune_data)
    final_result_filename_func = partial(val_search_combination_idx_final_test_filename_func, selected_idx)

    eval_results = query_with_predefined_shots(args, task_specific_helper, tune_selected_shots,
                        test_data, final_result_filename_func)
    print_tabular_results("GREEDY_ACC" if args.test_num_samples == 1 else "VOTE_ACC", eval_results)

def main():
    parser = argparse.ArgumentParser()
    register_base_args(parser)
    register_cons_args(parser)
    register_validation_args(parser)
    args = parser.parse_args()

    # post process args
    assert args.task is not None
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
