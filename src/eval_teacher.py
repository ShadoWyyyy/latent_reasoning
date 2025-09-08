# teacher_extract_sample.py
import os
import json
import numpy as np
import pickle
from tqdm import tqdm
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
from transformers import GPT2LMHeadModel, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_best_model(checkpoint_path, model_name="gpt2"):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    if os.path.exists(checkpoint_path):
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(checkpoint_path, map_location=map_location)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        print(f"✅ 模型已加载并移动到 {device}：{checkpoint_path}")
        return model
    else:
        raise FileNotFoundError(f"模型文件未找到：{checkpoint_path}")

model = load_best_model(
    "/run/determined/NAS1/public/chengjintao/teacher_checkpoints/0824_teacher_best_model.pt"
)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

train_path = "../data/gsm8k_aug_train.jsonl"
with open(train_path, "r", encoding="utf-8") as f:
    train_data = [json.loads(line) for line in f]

# 保存的结果
saved_tokens = []   # 正确答案对应的 token ids
saved_hiddens = []  # 正确答案对应的 last hidden states
meta_info = []      # 保存 question idx, sample idx 等元信息

num_samples = 10
temperature = 0.7
hidden_dim = model.config.hidden_size

for idx, ex in enumerate(tqdm(train_data, desc="Sampling")):
    q = ex["question"].strip()
    gt_answer = f"####{ex['answer'].strip()}{tokenizer.eos_token}"

    # prompt = 问题
    q_ids = tokenizer(q, return_tensors="pt").input_ids.to(device)

    # 一次性并行采样序列
    with torch.no_grad():
        gen_ids = model.generate(
            q_ids.repeat(num_samples, 1),  # [20, q_len]
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            max_new_tokens=128,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )  # shape [20, seq_len]

    # 解码结果
    decoded = tokenizer.batch_decode(gen_ids[:, q_ids.size(1):], skip_special_tokens=True)

    # 遍历每个采样，保留正确答案的 hidden states
    seen_sequences = set()  # 用来去重，存储 tuple(token_ids)

    for s_idx, (seq_ids, text) in enumerate(zip(gen_ids, decoded)):
        text = f"{text}{tokenizer.eos_token}"
        if gt_answer in text:  # 粗暴匹配答案正确
            seq_tuple = tuple(seq_ids.tolist())  # 转成不可变 tuple 用于比较
            if seq_tuple in seen_sequences:
                continue  # 已经保存过这个序列，跳过
            seen_sequences.add(seq_tuple)

            with torch.no_grad():
                out = model(seq_ids.unsqueeze(0), output_hidden_states=True)
                hs = out.hidden_states[-1][0].cpu().numpy()  # (seq_len, hidden_dim)

            saved_tokens.append(seq_ids.cpu().numpy())
            saved_hiddens.append(hs.astype(np.float32))
            meta_info.append({"q_idx": idx, "sample_idx": s_idx, "answer": gt_answer})

# 保存
OUT_PKL = "/run/determined/NAS1/public/chengjintao/saved_teacher_hiddens/0824_teacher_autoregressive_sampling.pkl"
data_to_save = {
    "tokens": saved_tokens,
    "hiddens": saved_hiddens,
    "meta": meta_info,
}

with open(OUT_PKL, "wb") as f:
    pickle.dump(data_to_save, f)

print(f"✅ Saved sampled results to {OUT_PKL}")
