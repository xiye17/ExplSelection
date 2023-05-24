import os
import itertools
import time
import openai

from tqdm import tqdm
from utils import *
from transformers import GPT2TokenizerFast

API_ERROR_IDENTIFIER = "OPENAI Error"
_TOKENIZER = GPT2TokenizerFast.from_pretrained("gpt2")
GPT3_LENGTH_LIMIT = 2049
GPT_MAX_ATTEMPTS = 60
GPT_WAITTIME = 10
API_ERROR_IDENTIFIER = "OPENAI Error"

DEFAULT_TRAIN_SEP = "\n\n"
DEFAULT_COMPLETION_LEADING = "\nA:"

def register_query_args(parser):
    parser.add_argument('--engine', default='code-davinci-002', choices=["davinci", "text-davinci-001", "text-davinci-002","text-davinci-003", "text-curie-001", "code-davinci-001",
        "code-davinci-002"])
    parser.add_argument('--run_prediction', default=False, action='store_true')
    parser.add_argument('--do_dryrun', default=False, action='store_true')
    parser.add_argument('--force_override', default=False, action='store_true')
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)

def register_base_args(parser):
    # standard, instruction, etc
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--slice_train', type=int, default=0)
    parser.add_argument('--num_train', type=int, default=128)
    parser.add_argument('--num_shots', type=int, default=8)
    parser.add_argument('--slice_dev', type=int, default=0)
    parser.add_argument('--num_dev', type=int, default=500)
    parser.add_argument('--do_print', default=False, action='store_true')
    # parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--num_eval_samples', type=int, default=-1)
    register_query_args(parser)

def config_args_and_api(args):
    if args.batch_size == -1:
        args.batch_size = 1
    openai.api_requestor.TIMEOUT_SECS = 60

    if args.engine in ["davinci", "text-davinci-001", "text-davinci-002", "text-davinci-003", "text-curie-001", "code-davinci-001", "code-davinci-002"]:
        openai.api_key = os.getenv("OPENAI_API_KEY")
    else:
        raise RuntimeError("Engine not supported")

def gpt_style_tokenize(x):
    return _TOKENIZER.tokenize(x)

def length_of_prompt(prompt, max_tokens):
    return len(_TOKENIZER.tokenize(prompt)) + max_tokens


def gpt_safe_completion(engine, prompts, temperature, max_tokens, logprobs=1, num_samples=1, echo=True, stop="\n"):
    last_exc = None
    for i in range(GPT_MAX_ATTEMPTS):
        try:
            return openai.Completion.create(engine=engine, prompt=prompts,
                temperature=temperature, max_tokens=max_tokens, logprobs=logprobs, n=num_samples, echo=echo, stop=stop)
        except openai.error.RateLimitError as e:
            last_exc = e
            print("\rWARNING: OPENAI Rate Error", last_exc, end="")
            time.sleep(GPT_WAITTIME)
        except openai.error.APIError as e:
            last_exc = e
            print("\rWARNING: OPENAI API Error", last_exc)
        except openai.error.Timeout as e:
            last_exc = e
            print("\rWARNING: OPENAI Timeout Error", last_exc)
        except openai.error.APIConnectionError as e:
            last_exc = e
            print("\rWARNING: OPENAI APIConnection Error", last_exc, end="")
        except openai.error.ServiceUnavailableError as e:
            last_exc = e
            print("\rWARNING: OPENAI Service Error", last_exc, end="")
    # make a fake response
    fake_choices = [
        [{
            "text": p + " OPENAI Error - " + str(last_exc),
            "API Error": True,
        }] * num_samples
        for p in prompts
    ]
    fake_choices = itertools.chain(*fake_choices)
    resp = {
        "choices": fake_choices
    }
    return resp

def batch_query_engine(args, prompts, max_tokens):
    predictions = []
    resps = gpt_safe_completion(engine=args.engine, prompts=prompts, temperature=args.temperature, max_tokens=max_tokens, logprobs=1, num_samples=args.num_samples, echo=True, stop='\n')

    resps = resps["choices"]
    # print("RESPS", resps, len(resps))
    # print("P", prompts, len(prompts))
    resps = [resps[(i * args.num_samples):(i * args.num_samples + args.num_samples)] for i in range(len(prompts))]
    # print(resps, len(resps))
    for prompt, resp in zip(prompts, resps):
        for pred in resp:
            pred["prompt"] = prompt
            if len(pred["text"]) > len(prompt):
                pred["text"] = pred["text"][len(prompt):]
            else:
                pred["text"] = " NULL"
            pred["completion_offset"] = len(prompt)

    return resps

# args
# prompts: 2d array
# cache_filename
# for gpt, assuming apis are pretty robust
def run_completion_tasks_with_cache(args, cache_fileneme, prompts_by_examples, max_tokens=0):
    assert isinstance(prompts_by_examples, list) and isinstance(prompts_by_examples[0], list) and isinstance(prompts_by_examples[0][0], str)
    if max_tokens == 0:
        assert args.num_samples == 1
    shape_records = [len(x) for x in prompts_by_examples]
    data_size = sum(shape_records)

    if os.path.exists(cache_fileneme):
        print("Cached Predictions Detected:", cache_fileneme)
        if args.force_override:
            print("Force Overriding Previous Predictions")
        else:
            return read_json(cache_fileneme)

    samples = list(itertools.chain(*prompts_by_examples))

    renewed_results = []
    prompt_lengths = []
    request_pool = []

    task_max_tokens = max_tokens
    for idx, prompt in enumerate(samples):
        if args.do_dryrun:
            response = length_of_prompt(prompt, task_max_tokens)
            print("-----------------------------------------")
            print(prompt)
            print("LEN", response)
            prompt_lengths.append(response)

        # add to request pool if no cached results, or error happened
        request_pool.append((idx, prompt))

    if args.do_dryrun:
        print(cache_fileneme)
        print('Total request', len(request_pool))
        print('MAX', max(prompt_lengths), 'COMP', task_max_tokens)
        return

    num_request, batch_size = len(request_pool), args.batch_size
    num_batch = (num_request + batch_size - 1) // batch_size
    # prediction loop, auto managing batching for OPT
    print("Num total request", num_request)
    for batch_idx in tqdm(range(num_batch), total=num_batch, desc="Querying"):
        batch_start = batch_idx * batch_size
        batch_end = batch_start + batch_size
        reqs = request_pool[batch_start: batch_end]

        idx_lists = [x[0] for x in reqs]
        prompts = [x[1] for x in reqs]
        responses = batch_query_engine(args, prompts, task_max_tokens)

        assert len(idx_lists) == len(responses)
        for i, resp in zip(idx_lists, responses):
            renewed_results.append(resp)

    print(cache_fileneme)
    # save
    # read un indexed dev
    assert len(renewed_results) == sum(shape_records)

    # group by example
    slice_start = 0
    renewed_cache = []
    for n in shape_records:
        renewed_cache.append(renewed_results[slice_start: slice_start + n])
        slice_start = slice_start + n

    dump_json(renewed_cache, cache_fileneme)
    return renewed_cache
