import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import numpy as np
import pickle

TOKENIZER = AutoTokenizer.from_pretrained("gpt2")
TOKENIZER.pad_token = TOKENIZER.eos_token

def encode_qra(question, reasoning_list, answer, tokenizer):
    """分别编码 question、reasoning、answer"""
    q_text = f"{question.strip()}"
    r_text = "||"
    if isinstance(reasoning_list, (list, tuple)):
        r_text += " ".join([step.strip() for step in reasoning_list])
    else:
        r_text += reasoning_list.strip()

    a_text = f"####{answer.strip()}{tokenizer.eos_token}"

    # print(a_text)

    q_ids = tokenizer(q_text, add_special_tokens=False)["input_ids"]
    r_ids = tokenizer(r_text, add_special_tokens=False)["input_ids"]
    a_ids = tokenizer(a_text, add_special_tokens=False)["input_ids"]

    # 最终 input_ids 拼接
    input_ids = q_ids + r_ids + a_ids

    return input_ids, q_ids, r_ids, a_ids


def pad_tensor(maxL, t, pad_value):
    """
    t: [seq_len] 或 [seq_len, hidden_dim]
    pad_value: scalar for 1D tensor, or 0 for hidden_dim padding
    """
    if t.dim() == 1:
        return torch.cat([t, torch.full((maxL - t.size(0),), pad_value, dtype=t.dtype)])
    elif t.dim() == 2:
        pad_shape = (maxL - t.size(0), t.size(1))
        return torch.cat([t, torch.full(pad_shape, pad_value, dtype=t.dtype)])
    else:
        raise ValueError("Unsupported tensor dimension for padding")


class CoTJSONLDataset(Dataset):
    def __init__(self, jsonl_path, max_len=512):
        self.jsonl_path = jsonl_path
        with open(jsonl_path, "r", encoding="utf-8") as f:
            self.examples = [json.loads(l) for l in f]
        self.max_len = max_len
        self.tokenizer = TOKENIZER

    def __len__(self):
        return len(self.examples)

    def truncate_teacher(self, q_ids, r_ids, a_ids):
        """优先保留 answer + reasoning(末尾) + question(末尾)"""
        if len(q_ids) + len(r_ids) + len(a_ids) <= self.max_len:
            return q_ids, r_ids, a_ids

        keep_a = a_ids
        max_r_len = max(0, self.max_len - len(keep_a))
        keep_r = r_ids[-max_r_len:] if len(r_ids) > max_r_len else r_ids

        max_q_len = max(0, self.max_len - len(keep_a) - len(keep_r))
        keep_q = q_ids[-max_q_len:] if len(q_ids) > max_q_len else q_ids

        return keep_q, keep_r, keep_a

    def __getitem__(self, idx):
        ex = self.examples[idx]
        q = ex["question"].strip()
        reasoning = ex.get("cot_steps", [])
        a = ex["answer"].strip()

        input_ids, q_ids, r_ids, a_ids = encode_qra(q, reasoning, a, self.tokenizer)

        attention_mask = torch.ones(len(input_ids), dtype=torch.long)
        labels = torch.full((len(input_ids),), -100, dtype=torch.long)

        q_len = len(q_ids)
        r_len = len(r_ids)
        a_len = len(a_ids)

        # 仅计算 reasoning 和 answer 部分
        labels[q_len:] = torch.tensor(input_ids[q_len:], dtype=torch.long)

        # reasoning mask (不含 question，含 terminator '||')
        r_mask = torch.zeros(len(input_ids), dtype=torch.bool)
        r_mask[q_len : q_len + r_len] = 1

        # answer mask (含 ####answer + eos)
        a_mask = torch.zeros(len(input_ids), dtype=torch.bool)
        a_mask[q_len + r_len :] = 1

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": attention_mask,
            "labels": labels,
            "r_mask": r_mask,
            "a_mask": a_mask,
        }

def collate_teacher(batch):
    maxL = max(x["input_ids"].size(0) for x in batch)
    return {
        "input_ids": torch.stack([pad_tensor(maxL, x["input_ids"], TOKENIZER.pad_token_id) for x in batch]),
        "attention_mask": torch.stack([pad_tensor(maxL, x["attention_mask"].float(), 0) for x in batch]),
        "labels": torch.stack([pad_tensor(maxL, x["labels"], -100) for x in batch]),
        "r_mask": torch.stack([pad_tensor(maxL, x["r_mask"].long(), 0) for x in batch]),
        "a_mask": torch.stack([pad_tensor(maxL, x["a_mask"].long(), 0) for x in batch]),
    }



class LRSDataset(Dataset):
    def __init__(self, json_path, teacher_pkl, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len

        # load data
        with open(json_path, "r", encoding="utf-8") as f:
            self.examples = [json.loads(l) for l in f]

        # load teacher hidden
        with open(teacher_pkl, 'rb') as f:
            data = pickle.load(f)

        teacher_as = data['teacher_as']
        teacher_rs = data['teacher_rs']
        r_lens = data['r_lens']

        self.teacher_reasoning_hiddens = teacher_rs
        self.teacher_reasoning_steps = r_lens
        self.teacher_answer_hiddens = teacher_as

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        q_text = f"{ex['question'].strip()}"
        reasoning_list = ex.get("cot_steps", [])
        r_text = "||"
        if isinstance(reasoning_list, (list, tuple)):
            r_text += " ".join([step.strip() for step in reasoning_list])
        else:
            r_text += reasoning_list.strip()
        a_text = f"####{ex['answer'].strip()}{self.tokenizer.eos_token}"

        q_ids = self.tokenizer(q_text, add_special_tokens=False)["input_ids"]
        r_ids = self.tokenizer(r_text, add_special_tokens=False)["input_ids"]
        a_ids = self.tokenizer(a_text, add_special_tokens=False)["input_ids"]

        input_ids = q_ids + r_ids + a_ids
        q_len = len(q_ids)
        a_len = len(a_ids)
        r_len = self.teacher_reasoning_steps[idx]

        assert len(r_ids) == r_len, f"Reasoning tokens length mismatch: len(r_ids)={len(r_ids)}, r_len={r_len}"

        # teacher hidden
        teacher_r_h = self.teacher_reasoning_hiddens[idx]
        teacher_a_h = self.teacher_answer_hiddens[idx]  # [ans_len, hidden_dim]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "teacher_reasoning_hiddens": torch.tensor(teacher_r_h, dtype=torch.float32),
            "teacher_answer_hiddens": torch.tensor(teacher_a_h, dtype=torch.float32),
            "q_len": q_len,
            "r_len": r_len,
            "a_len": a_len
        }


def collate_student(batch):
    maxL = max(x["input_ids"].size(0) for x in batch)
    input_ids = torch.stack([pad_tensor(maxL, x["input_ids"], TOKENIZER.pad_token_id) for x in batch])

    max_r = max(x["r_len"] for x in batch)
    max_a = max(x["a_len"] for x in batch)

    # teacher hidden padding
    teacher_reasoning_hiddens = torch.stack([pad_tensor(max_r, x["teacher_reasoning_hiddens"], 0.0) for x in batch])
    teacher_answer_hiddens = torch.stack([pad_tensor(max_a, x["teacher_answer_hiddens"], 0.0) for x in batch])

    q_lens = torch.tensor([x["q_len"] for x in batch], dtype=torch.long)
    r_lens = torch.tensor([x["r_len"] for x in batch], dtype=torch.long)
    a_lens = torch.tensor([x["a_len"] for x in batch], dtype=torch.long)

    return {
        "input_ids": input_ids,
        "teacher_reasoning_hiddens": teacher_reasoning_hiddens,
        "teacher_answer_hiddens": teacher_answer_hiddens,
        "q_len": q_lens,
        "r_len": r_lens,
        "a_len": a_lens
    }

