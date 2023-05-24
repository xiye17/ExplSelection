from utils import *

def load_train_test_set(args):
    train_data = read_json("data/{}_{}.json".format(args.task, "train"))
    test_data = read_json("data/{}_{}.json".format(args.task, "test"))
    train_data = train_data[args.slice_train:args.slice_train+args.num_train]
    test_data = test_data[args.slice_dev:args.slice_dev+args.num_dev]
    return train_data, test_data

class TaskHelper:
    style_to_completion_length = {}

    def __init__(self, style):
        self.style = style

    @classmethod
    def from_taskname(cls, taskname, style):
        if taskname == "gsm":
            return GSMTaskHelper(style)
        elif taskname == "ecqa":
            return ECQATaskHelper(style)
        elif taskname == "esnli":
            return ESNLITaskHelper(style)
        elif taskname == "strategyqa":
            return StrategyQATaskHelper(style)
        else:
            raise RuntimeError("Not Implemented Yet")

    def prompt_func(self, test_ex, shots):
        raise RuntimeError("Not Implemented Yet")

    def get_completion_length(self):
        return self.style_to_completion_length[self.style]


class GSMTaskHelper(TaskHelper):
    style_to_completion_length = {
        "default": 160,
        "standard": 32,
    }
    def prompt_func(self, test_ex, shots):
        if self.style == "default":
            return self.default_prompt(test_ex, shots)
        elif self.style == "standard":
            return self.standard_prompt(test_ex, shots)
        else:
            raise RuntimeError("Not Implemented Yet")

    def default_prompt(self, test_ex, shots):
        showcase_examples = [
            "Q: {}\nA: {}\n".format(x["question"], x["completion"]) for x in shots
        ]
        test_example = ["Q: {}\nA:".format(test_ex["question"])]
        return "\n".join(showcase_examples + test_example)

    def strip_explanation(self, comp):
        return comp[comp.find("The answer is"):]

    def standard_prompt(self, test_ex, shots):
        showcase_examples = [
            "Q: {}\nA: {}\n".format(x["question"], self.strip_explanation(x["completion"])) for x in shots
        ]
        test_example = ["Q: {}\nA:".format(test_ex["question"])]
        return "\n".join(showcase_examples + test_example)


class ECQATaskHelper(TaskHelper):
    style_to_completion_length = {
        "default": 96,
        "standard": 24,
    }

    def prompt_func(self, test_ex, shots):
        if self.style == "default":
            return self.default_prompt(test_ex, shots)
        elif self.style == "standard":
            return self.standard_prompt(test_ex, shots)
        else:
            raise RuntimeError("Not Implemented Yet")

    def strip_explanation(self, comp):
        return comp[comp.find("the answer is"):]

    def standard_prompt(self, test_ex, shots):
        showcase_examples = [
            "Q: {}\nA: {}\n".format(x["question"], self.strip_explanation(x["completion"])) for x in shots
        ]
        test_example = ["Q: {}\nA:".format(test_ex["question"])]
        return "\n".join(showcase_examples + test_example)

    def default_prompt(self, test_ex, shots):
        showcase_examples = [
            "Q: {}\nA: {}\n".format(x["question"], x["completion"]) for x in shots
        ]
        test_example = ["Q: {}\nA:".format(test_ex["question"])]
        return "\n".join(showcase_examples + test_example)


class ESNLITaskHelper(TaskHelper):
    style_to_completion_length = {
        "qaneither": 96,
        "qanotpossible": 96,
        "psource": 96,
        "psourcestandard": 16,
    }

    style_to_label_mapping = {
        "qaneither": {
            'entailment': 'true',
            'contradiction': 'false',
            'neutral': 'neither',
        },
        "qanotpossible": {
            'entailment': 'true',
            'contradiction': 'false',
            'neutral': 'not possible to tell',
        },
        "psource": {
            'entailment': 'yes',
            'contradiction': 'no',
            'neutral': 'not possible to tell',
        },
        "psourcestandard": {
            'entailment': 'yes',
            'contradiction': 'no',
            'neutral': 'not possible to tell',
        },
    }

    def __init__(self, style):
        self.style = style
        self.label_mapping = ESNLITaskHelper.style_to_label_mapping[style]

    def prompt_func(self, test_ex, shots):
        if self.style == "qaneither":
            return self.qaneither_prompt(test_ex, shots)
        elif self.style == "qanotpossible":
            return self.qanotpossible_prompt(test_ex, shots)
        elif self.style == "psource":
            return self.psource_prompt(test_ex, shots)
        elif self.style == "psourcestandard":
            return self.psourcestandard_prompt(test_ex, shots)
        else:
            raise RuntimeError("Not Implemented Yet")

    def psourcestandard_prompt(self, test_ex, shots):
        showcase_examples = [
            'Premise:\n"{}"\nBased on this premise, can we conclude the hypothesis "{}" is true?\nOPTIONS:\n- yes\n- no \n- not possible to tell\nA: The answer is {}.\n'.format(
                s["premise"], s["hypothesis"], self.label_mapping[s["label"]]) for s in shots
        ]
        input_example = 'Premise:\n"{}"\nBased on this premise, can we conclude the hypothesis "{}" is true?\nOPTIONS:\n- yes\n- no \n- not possible to tell\nA:'.format(test_ex["premise"], test_ex["hypothesis"])
        prompt = "\n".join(showcase_examples + [input_example])
        return prompt

    def qaneither_prompt(self, test_ex, shots):
        showcase_examples = [
            "{}\nQ: {} True, False, or Neither?\nA: {} The answer is {}.\n".format(s["premise"], s["hypothesis"], 
            s["explanations"][0], self.label_mapping[s["label"]]) for s in shots
        ]
        input_example = "{}\nQ: {} True, False, or Neither?\nA:".format(test_ex["premise"], test_ex["hypothesis"])
        prompt = "\n".join(showcase_examples + [input_example])
        return prompt

    def qanotpossible_prompt(self, test_ex, shots):
        showcase_examples = [
            "{}\nQ: {} True or False?\nA: {} The answer is {}.\n".format(s["premise"], s["hypothesis"], 
            s["explanations"][0], self.label_mapping[s["label"]]) for s in shots
        ]
        input_example = "{}\nQ: {} True, or False?\nA:".format(test_ex["premise"], test_ex["hypothesis"])
        prompt = "\n".join(showcase_examples + [input_example])
        return prompt

    def psource_prompt(self, test_ex, shots):
        showcase_examples = [
            'Premise:\n"{}"\nBased on this premise, can we conclude the hypothesis "{}" is true?\nOPTIONS:\n- yes\n- no \n- not possible to tell\nA: {} The answer is {}.\n'.format(
                s["premise"], s["hypothesis"], 
            s["explanations"][0], self.label_mapping[s["label"]]) for s in shots
        ]
        input_example = 'Premise:\n"{}"\nBased on this premise, can we conclude the hypothesis "{}" is true?\nOPTIONS:\n- yes\n- no \n- not possible to tell\nA:'.format(test_ex["premise"], test_ex["hypothesis"])
        prompt = "\n".join(showcase_examples + [input_example])
        return prompt


class StrategyQATaskHelper(TaskHelper):
    style_to_completion_length = {
        "default": 96,
        "stdqa": 96,
        "standard": 8,
    }

    def prompt_func(self, test_ex, shots):
        if self.style == "default":
            return self.default_prompt(test_ex, shots)
        elif self.style == "stdqa":
            return self.stdqa_prompt(test_ex, shots)
        elif self.style == "standard":
            return self.standard_prompt(test_ex, shots)
        else:
            raise RuntimeError("Not Implemented Yet")
    

    def standard_prompt(self, test_ex, shots):
        showcase_examples = [
            "Q: {}\nA: The answer is {}.\n".format(x["question"], x["answer"]) for x in shots
        ]
        test_example = ["Q: {}\nA:".format(test_ex["question"])]
        return "\n".join(showcase_examples + test_example)

    def stdqa_prompt(self, test_ex, shots):
        if shots:
            showcase_examples = [
                "Q: {}\nA: {} So the answer is {}.\n".format(x["question"], " ".join(x["facts"]), x["answer"]) for x in shots
            ]
        else:
            showcase_examples = []
        test_example = ["Q: {}\nA:".format(test_ex["question"])]
        return "\n".join(showcase_examples + test_example)

    def default_prompt(self, test_ex, shots):
        if shots:
            showcase_examples = [
                "Q: {}\nA: {}\n".format(x["question"], x["completion"]) for x in shots
            ]
            raise RuntimeError("Not implemented")
        else:
            showcase_examples = []
        test_example = ["Q: Yes or no: {}\nA:".format(test_ex["question"])]
        return "\n".join(showcase_examples + test_example)
