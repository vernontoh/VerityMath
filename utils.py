import copy
import signal
import torch
from typing import Any, Dict
import bitsandbytes as bnb
from dataclasses import dataclass
from transformers import PreTrainedTokenizer


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def find_all_linear_names(args, model):
    cls = (
        bnb.nn.Linear4bit
    )  # if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def tokenize_data(example_dict, tokenizer, is_train, max_length=1024):
    features = {"input_ids": [], "attention_mask": [], "source_len": []}

    for question, code, answer in zip(
        example_dict["question"],
        example_dict["generated_code_string"],
        example_dict["answer"],
    ):
        # Formatting source and answer text
        source_text = f"Question: {question}\n\nPython solution:\n"

        if is_train:
            answer_text = code
        else:
            answer_text = ""

        # Tokenize source and answer text
        source_res = tokenizer(source_text, return_attention_mask=False)
        answer_res = tokenizer(answer_text, return_attention_mask=False)

        # Get input ids by combining source and answer ids
        source_ids = source_res["input_ids"]
        answer_ids = answer_res["input_ids"]
        input_ids = source_ids + answer_ids
        source_length = len(source_ids)

        # Add eos token for training
        if is_train:
            input_ids = input_ids + [tokenizer.eos_token_id]

        # Truncate input ids if it is more than max length
        input_ids = input_ids[:max_length]
        attention_mask = [1] * len(input_ids)
        if source_length > max_length:
            source_length = max_length
        features["input_ids"].append(input_ids)
        features["attention_mask"].append(attention_mask)
        features["source_len"].append(source_length)

    return features


@dataclass
class PaddedDataCollator:
    tokenizer: PreTrainedTokenizer

    def __call__(self, features):
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        max_input_length = max(len(x["input_ids"]) for x in features)

        for feature in features:
            # Left padding for examples with length less than max input length
            input_ids = [self.tokenizer.eos_token_id] * (
                max_input_length - len(feature["input_ids"])
            ) + feature["input_ids"]

            # Set attention mask to zero for padding ids
            attention_mask = [0] * (
                max_input_length - len(feature["attention_mask"])
            ) + feature["attention_mask"]

            # Set source text and padding to -100 for our labels so we do not include it in our loss
            labels = [-100] * (
                max_input_length - len(feature["input_ids"]) + feature["source_len"]
            ) + feature["input_ids"][feature["source_len"] :]

            # Each batch will consist of input_ids, attention_mask and labels
            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(labels)

        batch["input_ids"] = torch.tensor(batch["input_ids"])
        batch["attention_mask"] = torch.tensor(batch["attention_mask"])
        batch["labels"] = torch.tensor(batch["labels"])
        return batch


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def timeout_handler(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


class GenericRuntime:
    GLOBAL_DICT = {}

    def __init__(self):
        self._global_vars = copy.copy(self.GLOBAL_DICT)

    def exec_code(self, code_piece: str) -> None:
        exec(code_piece, self._global_vars)

    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars)

    def run_code(self, code_gen: str, answer_expr, time_out: float = 10):
        snippet = code_gen.strip().split("\n")
        counter_code_snippet = [
            "class Counter(dict):",
            "    def __init__(self, iterable=None):",
            "        super().__init__()",
            "        if iterable is not None:",
            "            super().update(iterable)",
            "    def __add__(self, other):",
            "        result = Counter()",
            "        for elem, count in self.items():",
            "            if elem in other:",
            "                newcount = count + other[elem]",
            "            else:",
            "                newcount = count",
            "            result[elem] = newcount",
            "        for elem, count in other.items():",
            "            if elem not in self:",
            "                result[elem] = count",
            "        return result",
            "    def __sub__(self, other):",
            "        result = Counter()",
            "        for elem, count in self.items():",
            "            if elem in other:",
            "                newcount = count - other[elem]",
            "            else:",
            "                newcount = count",
            "            result[elem] = newcount",
            "        for elem, count in other.items():",
            "            if elem not in self:",
            "                result[elem] = 0 - count",
            "        return result",
            "    def __eq__(self, other):",
            "        return all(self.get(e, 0) == other.get(e, 0) for c in (self, other) for e in c)",
        ]

        updated_code_snippet = []

        start = False
        for snippet_line in snippet:
            if "def solution():" in snippet_line:
                updated_code_snippet.append("def solution():")
                start = True
                continue

            elif snippet_line.strip() == "":
                continue

            elif "return" == snippet_line.strip().split(" ")[0]:
                updated_code_snippet.append(snippet_line)
                break

            if start:
                updated_code_snippet.append(snippet_line)

        execute_code_gen = "\n".join(counter_code_snippet + updated_code_snippet)

        with timeout(time_out):
            try:
                self.exec_code(execute_code_gen)
                return updated_code_snippet, self.eval_code(answer_expr), None
            except AssertionError as ae:
                print("Assertion Error", flush=True)
                return updated_code_snippet, None, "Assertion Error"
            except BaseException as e:
                print(f"Code Excution Error: {e}", flush=True)
                return updated_code_snippet, None, "Code Execution Error"


class Counter(dict):
    def __init__(self, iterable=None):
        super().__init__()
        if iterable is not None:
            super().update(iterable)

    def __add__(self, other):
        result = Counter()
        for elem, count in self.items():
            if elem in other:
                newcount = count + other[elem]
            else:
                newcount = count
            result[elem] = newcount
        for elem, count in other.items():
            if elem not in self:
                result[elem] = count
        return result

    def __sub__(self, other):
        result = Counter()
        for elem, count in self.items():
            if elem in other:
                newcount = count - other[elem]
            else:
                newcount = count
            result[elem] = newcount
        for elem, count in other.items():
            if elem not in self:
                result[elem] = 0 - count
        return result

    def __eq__(self, other):
        return all(self.get(e, 0) == other.get(e, 0) for c in (self, other) for e in c)
