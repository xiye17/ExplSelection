import os
import argparse
import sys
import json
import re

import numpy as np

from abc import ABC
from collections import namedtuple, Counter, OrderedDict
from os.path import join


EVALUATOR_REGISTRY = {}

Prediction = namedtuple('Prediction', ['completion', 'prompt', 'logprob', 'norm_logprob'])


def print_tabular_results(row_id, eval_result):
    num_contents = [ "%.2f" % (eval_result["accuracy"] * 100), "%.2f" % (eval_result["consistency"] * 100),
        str(eval_result["avg_logprob"]), str(eval_result["avg_normlogprob"])]
    print("\t".join(["TABINFO", str(row_id)] + num_contents))

class TaskEvaluator(ABC):
    do_printing = False
    NULL_ANSWER = "null"

    @classmethod
    def get_task_name(cls):
        [task_name] = re.match("(.+)Evaluator", cls.__name__).groups()
        return task_name.lower()

    def __init_subclass__(cls, **kwargs):
        """Register all children in registry"""
        super().__init_subclass__(**kwargs)
        if cls == TaskEvaluator:
            # print(f"{cls} is abstract!")
            return
        task_name = cls.get_task_name().lower()
        EVALUATOR_REGISTRY[task_name] = cls

    @classmethod
    def process_instance(cls, pred, ref, prompting_style=None):
        choices = []
        gt = cls.post_process_ground_truth(ref["answer"])
        null_ans = cls.NULL_ANSWER
        prompt = pred[0].prompt
        for p in pred:
            single_comp, single_exp, single_ans = cls.parse_explanation_answer_from_completion(p.completion, prompting_style)
            choices.append({
                "completion": single_comp,
                "answer": single_ans,
                "explanation": single_exp,
                "norm_logprob": p.norm_logprob,
                "sum_logprob": p.logprob,
                "acc": str(gt == single_ans),
            })

        return {
            "prompt": prompt,
            "ground_truth": gt,
            "null_answer": null_ans,
            "completions": choices
        }
            

    @classmethod
    def evaluate(cls, predictions, examples, prompting_style=None, return_verbose=False):
        acc_records = []
        cov_records = []
        cons_records = []

        all_proced_answers = []
        all_proced_gts = []
        all_voted_answers = []
        for idx, (pred, ref) in enumerate(zip(predictions, examples)):
            if isinstance(pred, list):
                all_answers = []
                comp = []
                prompt = cls.post_process_prompt(pred[0].prompt)
                answer_counter = {}
                for p in pred:
                    single_comp, single_ans = cls.postprocess_completion(p.completion, prompting_style)
                    all_answers.append(single_ans)
                    comp.append(single_comp)
                    if single_ans not in answer_counter:
                        answer_counter[single_ans] = {
                            "count": 0,
                            "max_logprob": -1e6,
                            "max_norm_logprob": -1e6,
                        }
                    stat = answer_counter[single_ans]
                    stat["count"] = stat["count"] + 1
                    stat["max_logprob"] = max(stat["max_logprob"], p.logprob)
                    stat["max_norm_logprob"] = max(stat["max_norm_logprob"], p.norm_logprob)

                sorted_answers = sorted(answer_counter.keys(), key=lambda x: (answer_counter[x]["count"], answer_counter[x]["max_norm_logprob"]), reverse=True)
                # sorted_answers = sorted(answer_counter.keys(), key=lambda x: ( answer_counter[x]["max_norm_logprob"],answer_counter[x]["count"] ), reverse=True)
                answer = sorted_answers[0]
                if answer == "null" and len(sorted_answers) > 1:
                    answer = sorted_answers[1]
                if "null" in sorted_answers:
                    sorted_answers.remove("null")
                cons = answer_counter[answer]['count'] / len(pred)
                answer_counter = OrderedDict([(k, answer_counter[k]) for k in sorted_answers])
            else:
                prompt = cls.post_process_prompt(pred.prompt)
                comp, answer = cls.postprocess_completion(pred.completion, prompting_style)
                cons = 1.0
                answer_counter = None
                all_answers = [answer]

            gt = cls.post_process_ground_truth(ref["answer"])
            acc_records.append(answer == gt)
            cons_records.append(cons)
            if answer_counter is not None:
                cov_records.append(gt in answer_counter)
            all_proced_answers.append(all_answers)
            all_voted_answers.append(answer)
            all_proced_gts.append(gt)
            cls.print_instance_outputs(idx, prompt, comp, answer, gt, answer_counter)

        eval_results = {}
        acc_records = np.array(acc_records)
        print("ACC: {:.2f}".format(np.mean(acc_records) * 100))
        print("CONS: {:.2f}".format(np.mean(cons_records) * 100))
        eval_results["accuracy"] = np.mean(acc_records)
        eval_results["consistency"] = np.mean(cons_records)
        if cov_records:
            cov_records = np.array(cov_records)
            print("COV: {:.2f}".format(np.mean(cov_records) * 100))
            eval_results["converage"] = np.mean(cov_records)
        if return_verbose:
            eval_results["all_raw_predictions"] = all_proced_answers
            eval_results["all_gts"] = all_proced_gts
            eval_results["all_voted_predictions"] = all_voted_answers
            eval_results["acc_records"] = acc_records
        eval_results["num"] = len(acc_records)
        return eval_results

    @classmethod
    def print_instance_outputs(cls, idx, prompt, comp, answer, gt, answer_counter=None):
        if cls.do_printing:
            print("\n---------------------------------------------")
            print("Prompt:", prompt)
            if isinstance(comp, list):
                print("Completion:")
                for c in comp:
                    print("\t" + c.strip())
                if answer_counter:
                    print("\tCounter:", answer_counter)
            else:
                print("Completion:", comp.strip())
            print("Answer:", answer, " | GT:", gt)
            print("IDX", idx, "ACC:", answer == gt)
            if answer_counter:
                print("COV:", gt in answer_counter)

    @classmethod
    def core_evaluation(cls, predictions, examples, prompting_style=None):
        raise NotImplementedError()

    # process completion, return processed completion and answer
    @staticmethod
    def postprocess_completion(completion, prompting_style):
        raise NotImplementedError()

    @staticmethod
    def post_process_ground_truth(gt):
        raise NotImplementedError()

    @classmethod
    def parse_explanation_answer_from_completion(cls, completion, prompting_style):
        raise NotImplementedError()

    @staticmethod
    def post_process_prompt(prompt):
        return prompt.split("\n\n")[-1].strip()

class GSMEvaluator(TaskEvaluator):
    ANSWER_RE = re.compile(r"(\-?[0-9\.\,]+)")
    NULL_ANSWER = "null"
    ANSWER_HINT = "the answer is"

    @staticmethod
    def post_process_ground_truth(gt):
        return GSMEvaluator.extract_answer(gt).strip()

    @staticmethod
    def postprocess_completion(completion, prompting_style):
        completion = completion.rstrip().split("\n\n")[0]
        hint_sent = "the answer is"
        completion_lower = completion.lower()
        if hint_sent in completion_lower:
            answer = completion_lower.split(hint_sent)[1].rstrip(".").strip()
        else:
            answer = completion_lower
        numeric_answer = GSMEvaluator.extract_answer(answer).strip()
        return completion, numeric_answer

    @classmethod
    def process_instance(cls, pred, ref, prompting_style=None):
        if not isinstance(pred, list):
            pred = [pred]

        choices = []
        gt = cls.post_process_ground_truth(ref["answer"])
        raw_gt = ref["answer"]
        null_ans = cls.NULL_ANSWER
        prompt = pred[0].prompt
        for p in pred:
            single_comp, single_exp, single_ans = cls.parse_explanation_answer_from_completion(p.completion, prompting_style)
            ans = GSMEvaluator.extract_answer(single_ans).strip()
            choices.append({
                "completion": single_comp,
                "answer": ans,
                "raw_answer": single_ans,
                "explanation": single_exp,
                "norm_logprob": p.norm_logprob,
                "sum_logprob": p.logprob,
                "acc": str(gt == ans),
            })

        return {
            "prompt": prompt,
            "ground_truth": gt,
            "raw_ground_truth": raw_gt,
            "null_answer": null_ans,
            "completions": choices
        }

    @classmethod
    def parse_explanation_answer_from_completion(cls, completion, prompting_style):
        completion = completion.rstrip().split("\n\n")[0]
        hint_sent = "The answer is"
        # print(completion)
        if "The answer is" in completion:
            explanation = completion.split("The answer is")[0].strip()
            answer = completion.split("The answer is")[1].rstrip(".").strip()
        elif "the answer is" in completion:
            explanation = completion.split("the answer is")[0].strip()
            answer = completion.split("the answer is")[1].rstrip(".").strip()
        else:
            explanation = completion
            answer = GSMEvaluator.NULL_ANSWER
        return completion, explanation, answer

    @staticmethod
    def extract_answer(completion):
        match = GSMEvaluator.ANSWER_RE.search(completion)
        if match:
            match_str = match.group(0).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return GSMEvaluator.NULL_ANSWER


class ECQAEvaluator(TaskEvaluator):
    NULL_ANSWER = "null"
    ANSWER_HINT = "the answer is"

    @staticmethod
    def post_process_ground_truth(gt):
        return gt.strip()

    @staticmethod
    def postprocess_completion(completion, prompting_style):
        completion = completion.rstrip().split("\n\n")[0]
        hint_sent = "the answer is"
        completion_lower = completion.lower()
        if hint_sent in completion_lower:
            answer = completion_lower.split(hint_sent)[1].rstrip(".").strip()
        else:
            answer = ECQAEvaluator.NULL_ANSWER
        return completion, answer

class ESNLIEvaluator(TaskEvaluator):
    NULL_ANSWER = "null"
    ANSWER_HINT = "the answer is"

    @staticmethod
    def post_process_ground_truth(gt):
        return {
            'entailment': 'true',
            'contradiction': 'false',
            'neutral': 'neither',
        }[gt.strip()]

    @staticmethod
    def postprocess_completion(completion, prompting_style):
        completion = completion.rstrip().split("\n\n")[0]
        hint_sent = "the answer is"
        completion_lower = completion.lower()
        if hint_sent in completion_lower:
            answer = completion_lower.split(hint_sent)[1].rstrip(".").strip()
        else:
            answer = ECQAEvaluator.NULL_ANSWER

        if answer == "yes": answer = "true"
        if answer == "no": answer = "false"
        if answer == "not possible to tell": answer = "neither"
        return completion, answer


class StrategyQAEvaluator(TaskEvaluator):
    NULL_ANSWER = "null"
    ANSWER_HINT = "the answer is"

    @staticmethod
    def post_process_ground_truth(gt):
        return gt.strip()

    @staticmethod
    def postprocess_completion(completion, prompting_style):
        completion = completion.rstrip().split("\n\n")[0]
        hint_sent = "the answer is"
        completion_lower = completion.lower()
        if hint_sent in completion_lower:
            answer = completion_lower.split(hint_sent)[1].rstrip(".").strip()
        else:
            answer = ECQAEvaluator.NULL_ANSWER
        return completion, answer


def get_task_evaluator(taskname):
    return EVALUATOR_REGISTRY[taskname.lower()]
