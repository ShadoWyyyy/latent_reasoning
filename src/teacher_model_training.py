import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from dataset import *
from tqdm import tqdm

# reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

train_path = "../data/gsm8k_aug_train.jsonl"
test_path = "../data/gsm8k_aug_test.jsonl"

save_dir = "/run/determined/NAS1/public/chengjintao/teacher_checkpoints"
os.makedirs(save_dir, exist_ok=True)

# hyperpara
batch_size = 32
grad_accum_steps = 4
lr = 5e-4
warmup_ratio = 0.1
max_len = 512
max_epochs = 8

model_name = "gpt2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

train_dataset = CoTJSONLDataset(train_path, max_len=max_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_teacher, num_workers=2)
test_dataset = CoTJSONLDataset(test_path, max_len=max_len)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_teacher, num_workers=2)

optim = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

from torch.optim.lr_scheduler import LambdaLR
import numpy as np

total_steps = len(train_loader) // grad_accum_steps * max_epochs
warmup_steps = int(total_steps * warmup_ratio)
lr_min_factor = 0.1   # min_lr = 0.1 * base_lr

def lr_lambda(current_step: int):
    if current_step < warmup_steps:
        # 线性warmup: 0 -> 1
        return float(current_step) / float(max(1, warmup_steps))
    # 余弦衰减部分
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
    return lr_min_factor + (1 - lr_min_factor) * cosine_decay

sched = LambdaLR(optim, lr_lambda)

def evaluate(model, data_loader):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attn, labels=labels, return_dict=True)
            loss = outputs.loss
            losses.append(loss.item())
    model.train()
    return np.mean(losses)

raw_losses = []       # 每个 batch 的训练 loss
test_losses = []      # 每个 epoch 的测试 loss
train_losses = []     # 每个 epoch 的训练 loss
best_test_loss = float("inf")

for epoch in range(max_epochs):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    optim.zero_grad()

    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attn, labels=labels, return_dict=True)
        loss = outputs.loss

        raw_losses.append(loss.item())
        loss = loss / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()

        pbar.set_postfix({"loss": float(loss.item() * grad_accum_steps)})

    # 平均训练 loss
    avg_train_loss = np.mean(raw_losses[-len(train_loader):])
    train_losses.append(avg_train_loss)

    # ------------------------------
    # 在每个 epoch 结束时评估 test loss
    # ------------------------------
    avg_test_loss = evaluate(model, test_loader)
    test_losses.append(avg_test_loss)

    print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

    # ------------------------------
    # 保存模型 checkpoint
    # ------------------------------
    ckpt_path = os.path.join(save_dir, f"0824_teacher_epoch_{epoch}.pt")
    torch.save(model.state_dict(), ckpt_path)

    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        best_path = os.path.join(save_dir, "0824_teacher_best_model.pt")
        torch.save(model.state_dict(), best_path)
        print(f"✅ Best model updated at epoch {epoch}, saved to {best_path}")

print("Training finished.")

import matplotlib.pyplot as plt

losses = np.array(raw_losses)

# 设置滑动平均窗口大小（例如 50 个 batch）
window_size = 50
if len(losses) >= window_size:
    moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
else:
    moving_avg = losses  # 数据量太小时直接用原始 loss

fig, axs = plt.subplots(2, 2, figsize=(20, 12))

# 第一个子图: Batch级别的训练loss
axs[0, 0].plot(losses, label="Batch Train Loss", alpha=0.5)
axs[0, 0].plot(range(window_size-1, window_size-1 + len(moving_avg)), moving_avg, color='red', label=f"Moving Avg ({window_size})", linewidth=2)
axs[0, 0].set_xlabel("Batch")
axs[0, 0].set_ylabel("Loss")
axs[0, 0].set_title("Batch-level Training Loss")
axs[0, 0].legend()
axs[0, 0].grid(True)

# 第二个子图: Epoch级别的Train Loss
axs[0, 1].plot(np.arange(1, len(train_losses)+1)*len(train_loader), train_losses, marker="o", label="Epoch Train Loss", color="blue")
axs[0, 1].set_xlabel("Batch")
axs[0, 1].set_ylabel("Loss")
axs[0, 1].set_title("Epoch-level Training Loss")
axs[0, 1].legend()
axs[0, 1].grid(True)

# 第三个子图: Epoch级别的Test Loss
axs[1, 0].plot(np.arange(1, len(test_losses)+1)*len(train_loader), test_losses, marker="s", label="Epoch Test Loss", color="green")
axs[1, 0].set_xlabel("Batch")
axs[1, 0].set_ylabel("Loss")
axs[1, 0].set_title("Epoch-level Testing Loss")
axs[1, 0].legend()
axs[1, 0].grid(True)

axs[1, 1].axis('off')

# 调整子图之间的距离
plt.tight_layout()

# 保存图像
plt.savefig("teacher_model_fine_tuning_loss_curve_subplot.png")