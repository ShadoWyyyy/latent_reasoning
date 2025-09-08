from contextlib import contextmanager
import signal
import torch as th

# taken from
# https://stackoverflow.com/questions/492519/timeout-on-a-function-call
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)


def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            return eval(formula)
    except Exception as e:
        signal.alarm(0)
        print(f"Warning: Failed to eval {formula}, exception: {e}")
        return None


def use_calculator(sample):
    if "<<" not in sample:
        return None

    parts = sample.split("<<")
    remaining = parts[-1]
    if ">>" in remaining:
        return None
    if "=" not in remaining:
        return None
    lhs = remaining.split("=")[0]
    lhs = lhs.replace(",", "")
    if any([x not in "0123456789*+-/.()" for x in lhs]):
        return None
    return eval_with_timeout(lhs)


def sample(model, qn, tokenizer, device, sample_len):
    # Inefficient version of calculator sampling -- no batches, doesn't
    # cache activations from previous tokens
    EQUALS_TOKENS = set([28, 796, 47505])

    for _ in range(sample_len):
        with th.no_grad():
            toks = tokenizer([qn], padding=False, return_tensors="pt").to(device)
            orig_len = toks["input_ids"].shape[1]

            out = model.generate(
                **toks, 
                max_length=orig_len + 1, 
                pad_token_id=model.config.eos_token_id,
                do_sample=False,
            )
            text = tokenizer.batch_decode(out)[0]

            if out[0, -1].item() in EQUALS_TOKENS:
                answer = use_calculator(text)
                if answer is not None:
                    # print("Triggered calculator, answer", answer)
                    text = text + str(answer) + ">>"

            qn = text
    return qn


import json
import os
import re
import torch as th


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def get_examples(split):
    path = os.path.join("../data/", f"{split}.jsonl")
    examples = read_jsonl(path)

    for ex in examples:
        ex.update(question=ex["question"])
        ex.update(answer=ex["answer"])

    print(f"{len(examples)} {split} examples")
    return examples


ANS_RE = re.compile(r"####(\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_completion, gt_example):
    gt_answer = gt_example["answer"]
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer