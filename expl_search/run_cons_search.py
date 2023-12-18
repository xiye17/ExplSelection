import sys
sys.path.append('.')

import os
import argparse
import random
import numpy as np

from tqdm import tqdm
from functools import reduce

from utils import *
from api_utils import (
    run_completion_tasks_with_cache,
    config_args_and_api,
    register_base_args,
    DEFAULT_TRAIN_SEP,
    DEFAULT_COMPLETION_LEADING,
    gpt_style_tokenize
)

from run_manual import read_manual_prompt
from run_selective import (
    load_train_test_set,
    TaskHelper,
    run_evaluation
)

from task_evaluator import get_task_evaluator
from sklearn.cluster import AgglomerativeClustering, KMeans
import textdistance

MAX_SAMPLE_VARIATIONS = 8


def register_cons_args(parser):
    parser.add_argument('--style_template', type=str, default="default")

    parser.add_argument("--aug_engine", default=None, choices=['opt', "davinci", "text-davinci-001", "text-davinci-002", "text-curie-001", "code-davinci-001",
        "code-davinci-002"])
    parser.add_argument("--aug_num_samples", type=int, default=1)
    parser.add_argument("--aug_temperature", type=float, default=0.0)

    parser.add_argument("--aug_batch_size", type=int, default=1)
    parser.add_argument("--score_batch_size", type=int, default=10)

    parser.add_argument("--test_engine", default=None, choices=['opt', "davinci", "text-davinci-001", "text-davinci-002", "text-curie-001", "code-davinci-001",
        "code-davinci-002"])
    parser.add_argument("--test_num_samples", type=int, default=1)
    parser.add_argument("--test_temperature", type=float, default=0.0)

    parser.add_argument("--search_objective", type=str, default="maxmean", choices=["maxmean", "maxsum"])
    parser.add_argument("--randseed", default=0, type=int)

    parser.add_argument("--pool_strategy", type=str, default="edsg", choices=["edsg", "unpl", "random", ])
    parser.add_argument("--do_manual_prompt", default=False, action='store_true')
    parser.add_argument('--manual_prompt_id', type=str, default="default")

    parser.add_argument("--do_inspect", default=False, action='store_true')

def leave_one_result_filename_func(args):
    if args.do_manual_prompt:
        return "expl_search/misc/loaug--{}--map{}--sty{}--eng{}--numsamp{}--temp{}--predictions.json".format(args.task,
            args.manual_prompt_id,
            args.style_template,
            args.aug_engine,
            args.aug_num_samples,
            args.aug_temperature,
        )
    else:
        return "expl_search/misc/loaug--{}--tr{}-{}--rand{}--numshot{}--sty{}--eng{}--numsamp{}--temp{}--predictions.json".format(args.task,
            args.slice_train, args.slice_train + args.num_train,
            args.randseed,
            args.num_shots,
            args.style_template,
            args.aug_engine,
            args.aug_num_samples,
            args.aug_temperature, 
        )


def leave_one_all_candidate_info_filename_func(args):
    resp_file = leave_one_result_filename_func(args)
    resp_file = resp_file.replace("--predictions.json", "--all-candidate-info.json")
    # resp_file = resp_file.replace("misc/", "expl_search/misc/")
    return resp_file

def leave_one_candidate_pool_filename_funct(args):
    resp_file = leave_one_result_filename_func(args)
    resp_file = resp_file.replace("--predictions.json", "--plstg{}--predictions.json".format(args.pool_strategy))
    return resp_file

def leave_one_candidate_filename_func(args):
    resp_file = leave_one_candidate_pool_filename_funct(args)
    resp_file = resp_file.replace("--predictions.json", "--candidates.json")
    resp_file = resp_file.replace("expl_search/misc/", "expl_search/candidates/")
    return resp_file

def leave_one_candidate_score_filename_func(args):
    resp_file = leave_one_candidate_pool_filename_funct(args)
    resp_file = resp_file.replace("--predictions.json", "--pairscores.json")
    # resp_file = resp_file.replace("misc/", "expl_search/misc/")
    return resp_file

def leave_one_candidate_prompt_filename_func(args):
    resp_file = leave_one_candidate_pool_filename_funct(args)
    resp_file = resp_file.replace("--predictions.json", "--obj{}--finalprompt.json".format(args.search_objective))

    return resp_file

def optimized_prompt_query_result_filename_func(args):
    updated_suffix = "--dv{}-{}--searchobj{}--eng{}--numsamp{}--temp{}---predictions.json".format(
        args.slice_dev, args.slice_dev + args.num_dev,
        args.search_objective,
        args.test_engine,
        args.test_num_samples,
        args.test_temperature,
    )

    resp_file = leave_one_candidate_pool_filename_funct(args)
    resp_file = resp_file.replace("--predictions.json", updated_suffix)
    return resp_file

def get_completion_from_exampler(x):
    assert x.split("\n")[-1].startswith("A: ")
    return x.split("\n")[-1][3:]


def max_diff_selection(gt_completion, samples):
    # deduplication
    samples = remove_duplcate_samples(samples)
    if len(samples) + 1 <= MAX_SAMPLE_VARIATIONS:
        return samples

    if gt_completion.startswith("A: "):
        gt_completion = gt_completion[3:]
    num_to_select = MAX_SAMPLE_VARIATIONS - 1

    idxs = ["gt"] + [str(x["idx"]) for x in samples]    
    texts = [gt_completion] + [x["text"] for x in samples]

    from evaluate import load
    bertscore = load("bertscore")

    paired_scores = {}
    for p_id, pt in zip(idxs, texts):
        preds = [pt] * len(idxs)
        refs = texts
        results = bertscore.compute(predictions=preds, references=refs, lang="en")
        resuts = results["f1"]
        for res_id, ref_id in enumerate(idxs):
            paired_scores[(p_id, ref_id)] = resuts[res_id]


    selected_idx = ["gt"]
    for _ in range(num_to_select):
        min_max_sim = 10.0
        min_max_sim_id = None
        for id in idxs:
            if id in selected_idx:
                continue

            max_sim_to_selected = max([paired_scores[(id, x)] for x in selected_idx])
            if max_sim_to_selected < min_max_sim:
                min_max_sim = max_sim_to_selected
                min_max_sim_id = id
        selected_idx.append(min_max_sim_id)
    samples = [x for x in samples if str(x["idx"]) in selected_idx]
    return samples


def get_all_valid_responses(args, shots, responses):
    evaluator = get_task_evaluator(args.task)

    valid_samples = []
    for shot, resps_by_shot in zip(shots, responses):
        ex = {}
        question = shot.rsplit("\n", 1)[0]
        gt_completion = shot.rsplit("\n", 1)[-1]
        ex["question"] = question
        ex["completion"] = gt_completion
        samples = []
        _, shot_answer = evaluator.postprocess_completion(get_completion_from_exampler(shot), args.style_template)
        for i, r in enumerate(resps_by_shot):
            sample = {}
            sample["text"] = r["text"]

            _, comp_answer = evaluator.postprocess_completion(r["text"], args.style_template)
            # no need to look at bad answers
            if shot_answer != comp_answer:
                continue
            # sample id
            sample["idx"] = i
            samples.append(sample)
        ex["samples"] = samples
        valid_samples.append(ex)
    return valid_samples

def select_samples_by_clustering_over_edit_distance(args, gt_completion, samples):
    if len(samples) + 1 <= MAX_SAMPLE_VARIATIONS:
        return samples
    num_to_select = MAX_SAMPLE_VARIATIONS - 1
    num_all = len(samples)
    paired_dist = np.zeros((num_all, num_all))
    dist = textdistance.levenshtein
    token_per_sample = [gpt_style_tokenize(x["text"]) for x in samples]
    for i in range(num_all):
        paired_dist[i, i] = 0.0
        for j in range(i + 1, num_all):
            val = dist.distance(token_per_sample[i], token_per_sample[j]) / len(max(token_per_sample[i], token_per_sample[j], key=len))
            paired_dist[i, j] = val
            paired_dist[j, i] = val


    clustering = AgglomerativeClustering(metric="precomputed", n_clusters=num_to_select, linkage="single")
    clutster_ids = clustering.fit_predict(paired_dist)

    selected_samples = []
    for i in range(num_to_select):
        samples_in_cluster = [x for x, y in zip(samples, clutster_ids) if y == i]
        best_ppl_sample = max(samples_in_cluster, key=lambda x: x["pplx"])
        selected_samples.append(best_ppl_sample)
    return selected_samples

def select_samples_by_kmeans_over_pplx(args, gt_completion, samples):
    if len(samples) + 1 <= MAX_SAMPLE_VARIATIONS:
        return samples
    num_to_select = MAX_SAMPLE_VARIATIONS - 1
    num_all = len(samples)

    pplx_vals = [[x["pplx"]] for x in samples]
    clutster_ids = KMeans(n_clusters=num_to_select, random_state=42, n_init=10).fit_predict(pplx_vals)
    selected_samples = []
    for i in range(num_to_select):
        samples_in_cluster = [x for x, y in zip(samples, clutster_ids) if y == i]
        best_ppl_sample = max(samples_in_cluster, key=lambda x: x["pplx"])
        selected_samples.append(best_ppl_sample)
    return selected_samples


def make_json_file_for_candidate(args, shots, overgen_responses, filename):
    # adhoc compensation
    valid_respones = get_all_valid_responses(args, shots, overgen_responses)
    # score responses
    idxs_list = []
    score_pairs_list = []

    for i, ex in enumerate(valid_respones):
        q = ex["question"]
        gt = ex["completion"]
        idxs_list.append(f"{i}-gt")
        score_pairs_list.append(q+"\n" + gt)
        for s in ex["samples"]:
            idxs_list.append(f"{i}-{s['idx']}")
            score_pairs_list.append(q + "\nA:" + s["text"])
    print("Total number of candidates:", len(score_pairs_list))

    candidate_info_cache_name = leave_one_all_candidate_info_filename_func(args)
    prompts_to_complete = [[x] for x in score_pairs_list]

    if args.run_prediction:
        _saved_batch_size, _saved_temp, saved_ns = args.batch_size, args.temperature, args.num_samples
        args.num_samples = 1
        args.temperature = 0.0
        args.batch_size = args.score_batch_size

        score_responses = run_completion_tasks_with_cache(args, candidate_info_cache_name, prompts_to_complete, 0)
        args.batch_size = _saved_batch_size
        args.temperature = _saved_temp
        args.num_samples = saved_ns
    else:
        score_responses = read_json(candidate_info_cache_name)

    # score
    score_responses = flatten_nested_list([flatten_nested_list(x) for x in score_responses])
    assert len(score_responses) == len(score_pairs_list)


    idx_to_response = dict(zip(idxs_list, score_responses))
    samples_by_shots = []

    if args.pool_strategy == "random":
        for i, ex in enumerate(valid_respones):
            non_dup_samples = remove_duplcate_samples(ex["samples"])
            samples_by_shots.append(non_dup_samples[:(MAX_SAMPLE_VARIATIONS - 1)])
    elif args.pool_strategy == "unpl":
        for i, ex in enumerate(valid_respones):
            non_dup_samples = remove_duplcate_samples(ex["samples"])
            non_dup_samples = sorted(non_dup_samples, key=lambda x: score_of_pair_shot(args, idx_to_response[f"{i}-{x['idx']}"])["mean"], reverse=True)
            samples_by_shots.append(non_dup_samples[:(MAX_SAMPLE_VARIATIONS - 1)])
    elif args.pool_strategy == "edsg":
        for i, ex in enumerate(valid_respones):
            non_dup_samples = remove_duplcate_samples(ex["samples"])
            for s in non_dup_samples:
                s["pplx"] = score_of_pair_shot(args, idx_to_response[f"{i}-{s['idx']}"])["mean"]
            samples_by_shots.append(select_samples_by_clustering_over_edit_distance(args, ex["completion"], non_dup_samples))

    # to cope with legacy code
    meta = []
    for i, s in enumerate(samples_by_shots):
        ex = {}
        ex["question"] = valid_respones[i]["question"]
        ex["completion"] = valid_respones[i]["completion"]
        samples_by_shots = sorted(s, key=lambda x: int(x["idx"]))
        ex["samples"] = samples_by_shots
        meta.append(ex)
    dump_json(meta, filename, 1)

def remove_duplcate_samples(valid_samples):
    non_duplicate_sampels = []
    text_added = []
    for s in valid_samples:
        if s["text"] in text_added:
            continue
        non_duplicate_sampels.append(s)
        text_added.append(s["text"])
    return non_duplicate_sampels

def parse_candidate_file(args, filename, shots, responses):
    candidates = read_json(filename)

    assert len(candidates) == len(shots)

    valid_resps = []
    for (shot_i, ex_anno), shot, ex_resps in zip(enumerate(candidates), shots, responses):
        valid_resp_idx = [s["idx"] for s in ex_anno["samples"]]
        valid_resp_idx = sorted(valid_resp_idx)
        ex_valid_samples = [ex_resps[idx] for idx in valid_resp_idx]

        # handle duplication
        ex_valid_samples = remove_duplcate_samples(ex_valid_samples)
        ex_valid_samples = [shot.rsplit("\n", 1)[0] + "\nA:" + r["text"] for r in ex_valid_samples]
        ex_valid_samples = [shot] + ex_valid_samples
        valid_resps.append(ex_valid_samples)
    return valid_resps

def read_and_prepare_candidates(args, shots, responses):
    candidate_filename = leave_one_candidate_filename_func(args)
    if not os.path.exists(candidate_filename):
        make_json_file_for_candidate(args, shots, responses, candidate_filename)

    print("Candidate Explanations:", candidate_filename)
    valid_samples = parse_candidate_file(args, candidate_filename, shots, responses)
    return valid_samples

def leave_one_augment(args, shots, task_specific_helper):
    # TODO: dirty implementation for fast exp
    # sanity check
    assert shots[0].split("\n")[-1].startswith("A: ")

    task_max_tokens = task_specific_helper.get_completion_length()

    cache_filename = leave_one_result_filename_func(args)

    prompts_to_complete = []
    for i in range(args.num_shots):
        train_shots = [x for (j, x) in enumerate(shots) if j != i]
        test_shot = shots[i]
        test_shot = test_shot[:test_shot.find(DEFAULT_COMPLETION_LEADING) + len(DEFAULT_COMPLETION_LEADING)]
        prompt = DEFAULT_TRAIN_SEP.join(train_shots + [test_shot])
        prompts_to_complete.append([prompt])

    args.engine = args.aug_engine
    args.temperature = args.aug_temperature
    args.num_samples = args.aug_num_samples
    _saved_batch_size = args.batch_size
    args.batch_size = args.aug_batch_size
    responses = run_completion_tasks_with_cache(args, cache_filename, prompts_to_complete, task_max_tokens)
    responses = [flatten_nested_list(resps_by_example) for resps_by_example in responses]
    args.batch_size = _saved_batch_size

    return responses


def score_of_pair_shot(args, response, score_base=False):
    tokens = response["logprobs"]["tokens"]
    token_offset = response["logprobs"]["text_offset"]
    if score_base:
        completion_offset = response["prompt"].find(DEFAULT_COMPLETION_LEADING) + len(DEFAULT_COMPLETION_LEADING)
        completion_end_offset = response["prompt"].find(DEFAULT_TRAIN_SEP)

        completion_start_tok_idx = token_offset.index(completion_offset)
        completion_end_tok_idx = token_offset.index(completion_end_offset)
    else:
        completion_offset = response["prompt"].rfind(DEFAULT_COMPLETION_LEADING) + len(DEFAULT_COMPLETION_LEADING)

        completion_start_tok_idx = token_offset.index(completion_offset)
        completion_end_tok_idx = len(tokens)
    # return len(tokens) - completion_start_tok_idx

    tok_scores = response["logprobs"]["token_logprobs"][completion_start_tok_idx:completion_end_tok_idx + 1]
    toks = response["logprobs"]["tokens"][completion_start_tok_idx:completion_end_tok_idx + 1]

    # print("Core", toks)
    tok_scores = np.array(tok_scores)
    return {"sum": tok_scores.sum(), "mean": tok_scores.mean()}


def discrete_expl_search(args, candidates):
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

    prompt_cache_name = leave_one_candidate_prompt_filename_func(args)
    if os.path.exists(prompt_cache_name) and not args.force_override:
        return read_json(prompt_cache_name)

    score_cache_name = leave_one_candidate_score_filename_func(args)
    prompts_to_complete = [[x[1]] for x in pairs]

    if args.run_prediction:
        args.num_samples = 1
        args.temperature = 0.0
        _saved_batch_size = args.batch_size
        args.batch_size = args.score_batch_size

        responses = run_completion_tasks_with_cache(args, score_cache_name, prompts_to_complete, 0)
        args.batch_size = _saved_batch_size
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
        pair_idx_to_score[pair_idx] = score_of_pair_shot(args, resp)

    # select prompts by hard searching
    choice_generator = itertools.product(*[list(range(len(p))) for p in candidates])
    num_total_choice = reduce((lambda x, y: x * y), [len(p) for p in candidates])
    max_score = -1e10
    best_choice = None



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
        if score > max_score:
            max_score = score
            best_choice = choice

    selected_shots = [p[c_idx] for p, c_idx in zip(candidates, best_choice)]
    # cache_prompt
    dump_json(selected_shots, prompt_cache_name)
    return selected_shots

def query_with_predefined_shots(args, task_helper, shots, test_data, result_cache_name_func,
                                    query_args=None, return_verbose=False, return_responses=False):
    optimized_prompt = DEFAULT_TRAIN_SEP.join(shots)

    prompts_to_complete = []
    for test_ex in test_data:
        test_prompt = task_helper.prompt_func(test_ex, [])

        test_prompt = optimized_prompt + DEFAULT_TRAIN_SEP + test_prompt
        prompts_to_complete.append(
            [test_prompt]
        )

    task_max_tokens = task_helper.get_completion_length()
    cache_filename = result_cache_name_func(args)
    if query_args is None:
        query_args = {
            "engine": args.test_engine,
            "num_samples": args.test_num_samples,
            "temperature": args.test_temperature,
            "batch_size": args.batch_size,
        }
    _engine, _num_samples, _temperature, _batch_size = args.engine, args.num_samples, args.temperature, args.batch_size
    args.engine, args.num_samples, args.temperature, args.batch_size = query_args["engine"], query_args["num_samples"], query_args["temperature"], query_args["batch_size"]

    responses = run_completion_tasks_with_cache(args, cache_filename, prompts_to_complete, task_max_tokens)
    args.engine, args.num_samples, args.temperature, args.batch_size = _engine, _num_samples, _temperature, _batch_size

    responses = [flatten_nested_list(resps_by_example) for resps_by_example in responses]

    eval_results = run_evaluation(args, test_data, responses, return_verbose=return_verbose)
    if not return_responses:
        return eval_results
    else:
        return eval_results, responses

def get_random_shots(rand_seed, train_data, num_shots):
    np.random.seed(rand_seed)
    # make it consistent with what's in run_selective
    randome_scores = np.random.rand(len(train_data))

    selected_idx = sorted(enumerate(randome_scores), key=lambda x: x[1], reverse=True)[:num_shots]
    selected_idx = [x[0] for x in selected_idx]
    selected_shots = [train_data[i] for i in selected_idx]

    return selected_shots

def evaluate_overgen_responses(args, shots, responses):
    evaluator = get_task_evaluator(args.task)

    covered = []
    for shot, resps_by_shot in zip(shots, responses):
        ex = {}
        question = shot.rsplit("\n", 1)[0]
        gt_completion = shot.rsplit("\n", 1)[-1]
        ex["question"] = question
        ex["completion"] = gt_completion
        samples = []
        _, shot_answer = evaluator.postprocess_completion(get_completion_from_exampler(shot), args.style_template)

        cnt = 0
        for i, r in enumerate(resps_by_shot):
            sample = {}
            sample["text"] = r["text"]

            _, comp_answer = evaluator.postprocess_completion(r["text"], args.style_template)
            # no need to look at bad answers
            if shot_answer != comp_answer:
                continue
            cnt += 1
        covered.append(cnt)
    print("Covered", covered)

def consistency_search_exp_pipeline(args):
    train_data, test_data = load_train_test_set(args)
    task_specific_helper = TaskHelper.from_taskname(args.task, args.style_template)

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
        # for shots in valid_shots:
        #     print("-----------")
        #     for s in shots:
        #         print(s)
    optimized_shots = discrete_expl_search(args, valid_shots)
    if args.do_inspect:
        for orig_shot, opt_shot in zip(shots, optimized_shots):
            q = orig_shot.split(DEFAULT_COMPLETION_LEADING)[0]
            print("\n-----------")
            print(q)
            print("ORI:", orig_shot.split(DEFAULT_COMPLETION_LEADING)[1])
            print("OPT:", opt_shot.split(DEFAULT_COMPLETION_LEADING)[1])
    query_with_predefined_shots(args, task_specific_helper, optimized_shots, test_data, optimized_prompt_query_result_filename_func)

def main():
    parser = argparse.ArgumentParser()
    register_base_args(parser)
    register_cons_args(parser)
    args = parser.parse_args()

    assert args.task is not None

    if args.aug_engine is None:
        args.aug_engine = args.engine
    if args.test_engine is None:
        args.test_engine = args.engine

    config_args_and_api(args)
    consistency_search_exp_pipeline(args)

if __name__=='__main__':
    main()
