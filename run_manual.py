import os
import argparse
import itertools
from random import choices

from tqdm import tqdm
from math import ceil

import numpy as np

from utils import *
from api_utils import (
    run_completion_tasks_with_cache,
    config_args_and_api,
    register_base_args,
    DEFAULT_TRAIN_SEP
)

from task_helper import TaskHelper, load_train_test_set
from run_selective import run_evaluation
from task_evaluator import TaskEvaluator, get_task_evaluator, Prediction, print_tabular_results

def register_manual_args(parser):
    parser.add_argument('--manual_prompt_id', type=str, default=None, required=True)
    parser.add_argument('--style_template', type=str, default="default")

def manual_query_result_filename_func(args):
    return "misc/{}--eng{}--dv{}-{}--manual{}--numsamp{}--temp{}--sty{}--predictions.json".format(
        args.task,
        args.engine,
        args.slice_dev, args.slice_dev + args.num_dev,
        args.manual_prompt_id,
        args.num_samples,
        args.temperature,
        args.style_template
    )

def read_manual_prompt(task, prompt_id, style_template):    
    prompt_lines = read_jsonline(f'prompts/{task}.jsonline')
    d = dict([(x["id"], x) for x in prompt_lines])
    selected = d[prompt_id]
    if "style_template" in selected:
        assert selected["style_template"] == style_template
    return selected["prompt"]

def predict_framework(args):
    train_data, test_data = load_train_test_set(args)
    task_helper = TaskHelper.from_taskname(args.task, args.style_template)

    base_manual_prompt = read_manual_prompt(args.task, args.manual_prompt_id, args.style_template)
    prompts_to_complete = []    
    for test_ex in test_data:
        test_part = task_helper.prompt_func(test_ex, [])
        
        prompts_to_complete.append(
            [base_manual_prompt + DEFAULT_TRAIN_SEP + test_part]
        )

    task_max_tokens = task_helper.get_completion_length()
    cache_filename = manual_query_result_filename_func(args)
    responses = run_completion_tasks_with_cache(args, cache_filename, prompts_to_complete, task_max_tokens)
    responses = [flatten_nested_list(resps_by_example) for resps_by_example in responses]

    eval_results = run_evaluation(args, test_data, responses)
    print_tabular_results("VOTE"+str(args.num_eval_samples), eval_results)

def eval_framework(args):
    _, test_data = load_train_test_set(args)
    responses = read_json(manual_query_result_filename_func(args))
    responses = [flatten_nested_list(resps_by_example) for resps_by_example in responses]
    eval_results = run_evaluation(args, test_data, responses)
    print_tabular_results("VOTE"+str(args.num_eval_samples), eval_results)

def main():
    parser = argparse.ArgumentParser()
    register_base_args(parser)
    register_manual_args(parser)

    args = parser.parse_args()
    assert args.task is not None
    assert args.manual_prompt_id is not None

    config_args_and_api(args)
    if args.run_prediction:
        predict_framework(args)
    else:
        eval_framework(args)

if __name__=="__main__":
    main()
