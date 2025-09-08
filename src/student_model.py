import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from transformers import GPT2LMHeadModel, GPT2Config


class GPT2LatentReasoningStudent(nn.Module):
    """
    训练（有教师）：
      - 将 Q(wte) 与 教师 R hidden（投到学生维度） 和 A(wte) 拼接成 [Q, R_teacher, A]，
        送入 GPT-2
      - 只计算答案段 NLL（标准 shift）；
      - 只对答案段 hidden 与教师答案 hidden 做 MSE(mean) 一致性；R 段不参与损失。

    推理（无教师）：
      - 用 GPT-2 迭代生成一段“潜推理序列” H_R（长度由 r_lens 决定），
        作为 [Q, R_latent] 前缀，再自回归生成答案。
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        teacher_hidden_size: Optional[int] = 768,   # 教师 hidden 维度；None=与 student 相同
        alpha_consistency: float = 0.001,             # 答案段一致性损失系数
    ):
        super().__init__()
        # self.gpt2_lm: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(model_name)

        if os.path.isdir(model_name) or os.path.exists(os.path.join(model_name, "config.json")):
            self.gpt2_lm = GPT2LMHeadModel.from_pretrained(model_name)

        elif model_name.endswith(".pt") or model_name.endswith(".bin"):
            if not os.path.exists(model_name):
                raise FileNotFoundError(f"Weight file not found: {model_name}")

            map_location = "cuda" if torch.cuda.is_available() else "cpu"
            state_dict = torch.load(model_name, map_location=map_location)

            # 使用标准 GPT-2 配置初始化模型
            config = GPT2Config.from_pretrained("gpt2")  # 或你训练时用的配置
            self.gpt2_lm = GPT2LMHeadModel(config)
            self.gpt2_lm.load_state_dict(state_dict, strict=False)  # 允许部分加载

            print(f"Loaded state_dict from {model_name}")

        else:
            self.gpt2_lm = GPT2LMHeadModel.from_pretrained(model_name)

        self.transformer = self.gpt2_lm.transformer
        self.wte = self.transformer.wte
        self.config = self.gpt2_lm.config
        self.n_embd: int = self.config.n_embd
        self.vocab_size: int = self.config.vocab_size

        # 语言头
        # self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)
        self.lm_head = self.gpt2_lm.lm_head

        # teacher->student 维度对齐
        t_dim = teacher_hidden_size if teacher_hidden_size is not None else self.n_embd
        if t_dim == self.n_embd:
            self.teacher_to_student = nn.Identity()
        else:
            self.teacher_to_student = nn.Linear(t_dim, self.n_embd, bias=False)

        self.alpha_consistency = float(alpha_consistency)
        self.beta_nll = 1

        self.terminator_id = None
        self.exit_threshold = 1

    # ------------------------------------------------------------
    # 基础积木
    # ------------------------------------------------------------
    def _run_blocks(self, hidden_states: torch.Tensor, attention_bias: Optional[torch.Tensor]) -> torch.Tensor:
        """
        将一段序列 hidden 过 GPT-2 blocks。
        hidden_states: [B, L, D]
        attention_bias: [B, 1, 1, L] 或 None（加到注意力 logits）
        返回: [B, L, D]
        """
        x = hidden_states
        for blk in self.gpt2.h:
            x = blk(x, layer_past=None, attention_mask=attention_bias, use_cache=False)[0]
        return self.gpt2.ln_f(x)
    
    def _forward_from_embeds(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        # inputs_embeds: [B, L, D]
        out = self.gpt2_lm(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        return out.hidden_states[-1], out.logits  # [B,L,D], [B,L,V]

    # ------------------------------------------------------------
    # 训练拼装：[Q_embed, R_teacher, A_embed]
    # ------------------------------------------------------------
    def _build_QRA_train(
        self,
        input_ids: torch.Tensor,                 # [B, S] = Q + A
        q_len: torch.Tensor,                     # [B]
        r_len: torch.Tensor,                     # [B]
        a_len: torch.Tensor,                     # [B]
        teacher_reasoning_hiddens: torch.Tensor  # [B, R_t, Tdim]
    ):
        device = input_ids.device
        B, _ = input_ids.shape
        D = self.n_embd

        H_all, lengths = [], []
        for i in range(B):
            ql = int(q_len[i])
            rl = int(r_len[i])
            al = int(a_len[i])

            q_ids = input_ids[i, :ql]
            a_ids = input_ids[i, ql + rl: ql+ rl + al]
            r_hid = teacher_reasoning_hiddens[i, :rl]

            H_Q = self.wte(q_ids) if ql > 0 else input_ids.new_zeros(0, D)
            H_R = self.teacher_to_student(r_hid)  # [rl, D]
            H_A = self.wte(a_ids) if al > 0 else input_ids.new_zeros(0, D)
            # wte + hidden_state + wte

            H_seq = torch.cat([H_Q, H_R, H_A], dim=0)  # [L_i, D]
            H_all.append(H_seq)
            lengths.append(H_seq.shape[0])

        max_len = max(lengths)
        H_pad = input_ids.new_zeros(B, max_len, D, dtype=torch.float)

        # 填充 H_pad
        for i, H_seq in enumerate(H_all):
            H_pad[i, :lengths[i]] = H_seq

        attn_mask = torch.zeros(B, max_len, device=device)
        for i, length in enumerate(lengths):
            attn_mask[i, :length] = 1.0
        pad_bias = (1.0 - attn_mask) * -1e4

        return H_pad, attn_mask, lengths
    
    # ------------------------------------------------------------
    # token损失
    # ------------------------------------------------------------
    def _compute_nll_loss(self, logits, input_ids, q_len, r_len, a_len):
        """
        Compute NLL for answer tokens (vectorized, handles autoregressive shift).
        logits: [B, L, V]   (model output)
        input_ids: [B, L]   (original input token ids: Q + R + A)
        q_len, r_len, a_len: tensors [B] with lengths (int)
        Returns:
            scalar loss (mean over selected tokens)
        """
        B, L, V = logits.size()
        device = logits.device

        if L < 2:
            return logits.new_zeros(())

        logits_shift = logits[:, :-1, :].contiguous()    # [B, L-1, V]
        labels_shift = input_ids[:, 1:].contiguous()     # [B, L-1]

        # positions for labels in original indexing: [1, 2, ..., L-1]
        pos_labels = torch.arange(1, L, device=device).unsqueeze(0).expand(B, L-1)  # [B, L-1]

        # answer tokens in original indexing start at a_start = q_len + r_len
        a_start = (q_len + r_len - 1).unsqueeze(1)   # [B, 1]
        a_end = (q_len + r_len + a_len - 1).unsqueeze(1)  # exclusive end [B,1]

        # mask where labels correspond to answer tokens
        answer_mask = (pos_labels >= a_start) & (pos_labels < a_end)  # [B, L-1], boolean

        # Build masked labels: non-answer positions -> ignore_index (-100)
        labels_masked = labels_shift.clone()
        labels_masked[~answer_mask] = -100

        # Flatten and compute cross_entropy with ignore_index
        loss_nll = F.cross_entropy(
            logits_shift.view(-1, V),
            labels_masked.view(-1),
            ignore_index=-100,
            reduction="mean",
        )
        return loss_nll

    # ------------------------------------------------------------
    # 一致性损失
    # ------------------------------------------------------------
    def _compute_consistency(
        self,
        hs: torch.Tensor,                     # [B, L, D] 学生 hidden
        start: torch.Tensor,                  # [B] 每个样本的起始位置
        length: torch.Tensor,                 # [B] 每个样本的有效长度
        teacher_hiddens: torch.Tensor         # [B, Tmax, Tdim]
    ) -> tuple[torch.Tensor, int]:
        """
        通用一致性计算 (answer/reasoning 都能用)

        返回:
            cons_sum: torch.Tensor 标量, 总 loss
            cons_cnt: int 有效 token 数
        """
        if teacher_hiddens is None:
            return hs.new_zeros(()), 0

        B, L, D = hs.size()
        Tmax = teacher_hiddens.size(1)

        # 投影到 student hidden space
        tea_h = self.teacher_to_student(teacher_hiddens).detach()  # [B, Tmax, D]

        # 找到 batch 内最大长度
        max_len = int(length.max().item())
        if max_len == 0:
            return hs.new_zeros(()), 0

        # === 批量收集 student hidden ===
        stu_h = torch.zeros(B, max_len, D, device=hs.device)
        for i in range(B):
            li = int(length[i].item())
            if li > 0:
                stu_h[i, :li] = hs[i, start[i]: start[i] + li]

        # teacher 对齐
        tea_h = tea_h[:, :max_len, :]

        # 有效 mask
        mask_pad = torch.arange(max_len, device=hs.device).unsqueeze(0) < length.unsqueeze(1)  # [B, max_len]

        # === 计算 MSE（只取有效 token） ===
        mse = nn.MSELoss(reduction="sum")
        cons_sum = mse(stu_h[mask_pad], tea_h[mask_pad])   # [N, D] 对齐比较
        cons_cnt = mask_pad.sum().item()                   # 有效 token 数

        return cons_sum, cons_cnt

    # ------------------------------------------------------------
    # 训练 forward
    # ------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,                        # [B, S] = Q + A
        q_len: torch.Tensor,                            # [B]
        r_len: torch.Tensor,                            # [B]
        a_len: torch.Tensor,                            # [B]
        teacher_reasoning_hiddens: Optional[torch.Tensor] = None,  # [B,R_t,Tdim] (仅作为输入)
        teacher_answer_hiddens: Optional[torch.Tensor] = None,     # [B,Amax,Tdim]
        return_hidden: bool = False,
    ) -> Dict[str, Any]:
        if teacher_reasoning_hiddens is None:
            raise ValueError("训练时必须提供 teacher_reasoning_hiddens（仅作为中间块，不参与loss）。")

        # 1) 构造 [Q_embed, R_teacher, A_embed]
        H_in, attn_mask, _ = self._build_QRA_train(
            input_ids, q_len, r_len, a_len, teacher_reasoning_hiddens
        )  # H_in: [B,L,D]

        # 2) 把input_embeds传给lmheadmodel
        hs, logits = self._forward_from_embeds(H_in, attn_mask)

        # 3) 仅算答案段 NLL
        loss_nll = self._compute_nll_loss(logits, input_ids, q_len, r_len, a_len)

        # 4) 一致性
        # B = batch size
        a_start = q_len + r_len - 1   # [B]
        r_start = q_len - 1           # [B]

        # 一致性损失
        loss_cons_a, count_cons_a = self._compute_consistency(hs, a_start, a_len, teacher_answer_hiddens)
        loss_cons_r, count_cons_r = self._compute_consistency(hs, r_start, r_len, teacher_reasoning_hiddens)

        # 总一致性损失
        loss_cons = (loss_cons_a + loss_cons_r) / (count_cons_a + count_cons_r)

        # 5) 总 loss
        loss = self.beta_nll * loss_nll + self.alpha_consistency * loss_cons

        out = {
            "loss": loss,
            "loss_nll": loss_nll,
            "loss_consistency": loss_cons,
            "logits": logits,
            "hidden_states": hs,
        }
        if return_hidden:
            out["last_hidden_state"] = hs
        return out

    # ------------------------------------------------------------
    # 推理（批量）：先用 block 迭代出 R，再自回归生成 A
    # ------------------------------------------------------------
    @torch.no_grad()
    def generate_answer(
        self,
        q_input_ids: torch.Tensor,  # [1, Q_len]
        r_len_max: int = 8,
        max_new_tokens: int = 64,
        output_latent: bool = False,
    ):
        device = q_input_ids.device
        assert q_input_ids.size(0) == 1, "只支持单样本，请用generate_answer_batch做批量"

        # ==== 初始化 ====
        H_cur = self.wte(q_input_ids)  # [1,Q,D]
        generated = []
        latent_tokens = [] if output_latent else None
        eos_id = self.config.eos_token_id if self.config.eos_token_id is not None else -1
        terminator_id = self.terminator_id

        # ==== 阶段1：潜推理 ====
        term_found = False
        for _ in range(r_len_max):
            attn = torch.ones(1, H_cur.size(1), device=device)
            hs, logits = self._forward_from_embeds(H_cur, attn)  # [1, L, D]
            h_new = hs[:, -1:, :]                                 # [1,1,D]
            H_cur = torch.cat([H_cur, h_new], dim=1)

            probs = F.softmax(logits[:, -1, :], dim=-1)  # [1,V]
            next_token = torch.argmax(probs, dim=-1)     # [1]
            token_id = next_token.item()
            token_prob = probs[0, token_id].item()

            if output_latent:
                latent_tokens.append(int(token_id))  # 记录潜推理阶段token

            if (token_id == terminator_id) & (token_prob > self.exit_threshold):
                tok = torch.tensor([terminator_id], device=device)
                term_emb = self.wte(tok).unsqueeze(1)  # [1,1,D]
                H_cur[:, -1:, :] = term_emb
                generated.append(int(terminator_id))
                term_found = True
                break

        # 没命中：强制在进入答案阶段前追加 terminator 
        if not term_found: 
            tok = torch.tensor([terminator_id], device=device) 
            term_emb = self.wte(tok).unsqueeze(1) # [1,1,D] 
            H_cur = torch.cat([H_cur, term_emb], dim=1) 
            generated.append(int(terminator_id))

        # ==== 阶段2：答案 ====
        for _ in range(max_new_tokens):
            attn = torch.ones(1, H_cur.size(1), device=device)
            hs, logits = self._forward_from_embeds(H_cur, attn)   # [1, L, D]
            h_new = hs[:, -1:, :]                                 # [1,1,D]
            probs = F.softmax(logits[:, -1, :], dim=-1)           # [1,V]
            next_token = torch.argmax(probs, dim=-1)
            token_id = next_token.item()
            generated.append(token_id)

            tok_emb = self.wte(next_token).unsqueeze(1)
            H_cur = torch.cat([H_cur, tok_emb], dim=1)

            if eos_id >= 0 and token_id == eos_id:
                break

        if output_latent:
            return torch.LongTensor(latent_tokens + generated)
        else:
            return torch.LongTensor(generated), term_found
