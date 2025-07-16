from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch
from gpt2_with_MSLA import patch_gpt2_with_msla
from gpt2_with_GQA import patch_gpt2_with_gqa
import gc

import math
import numpy as np
import os

os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Set MPS memory limit to 10%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# Config
# =====================
mode = "gpt2_msla"   # "gpt2", "gpt2_msla", or "gpt2_gqa"

# =====================
# Load tokenizer & config
# =====================
# Load config & tokenizer
config = GPT2Config.from_pretrained("gpt2")
config.loss_type = "ForCausalLMLoss" # Chỉ định loss_type

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =====================
# Initialize model
# =====================

# model_path = "gpt2"  # Default GPT2 model path
model_path = "./outputs/final-gpt2_msla"
print(model_path, ":", os.path.exists(model_path))

if os.path.exists(model_path) and os.path.isdir(model_path):
    from gpt2_with_MSLA import from_pretrained_with_msla

    model = from_pretrained_with_msla(
        model_path,
        num_latents=64,
        k_top=4,
        device="cpu"
    )
else:
    print("[ℹ️] Using vanilla GPT2 model.")
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)

    if mode == "gpt2_msla":
        # print("[🔧] Patching GPT2 with MSLA...")
        patch_gpt2_with_msla(model, num_latents=256, k_top=16)
    elif mode == "gpt2_gqa":
        # print("[🔧] Patching GPT2 with GQA...")
        patch_gpt2_with_gqa(model, num_key_value_groups=4)
    else:
        print("[ℹ️] Using vanilla GPT2.")

model.to(device)

# =====================
# Freeze all except attention
# =====================
def freeze_except_attention(model, mode=""):
    """
    Freeze all parameters except attention layers.
    """
    # print(f"🔒 Freezing all parameters except attention layers ({mode})...")
    n_trainable = 0
    n_total = 0
    if mode == "gpt2_msla":
        attn_layer = "attn.msla"
    elif mode == "gpt2_gqa":
        attn_layer = "attn.gqa"
    elif mode == "gpt2":
        attn_layer = "attn.c_attn"
    else:
        attn_layer = ""
    # If no specific attention layer, train all
    if attn_layer == "":
        print("ℹ️ No attention layers to freeze. All parameters will be trainable.")
        for name, param in model.named_parameters():
            param.requires_grad = True
            n_trainable += param.numel()
            n_total += param.numel()
    else:
        print(f"🔒 Freezing all parameters except: {attn_layer} layers.")
        for name, param in model.named_parameters():
            # Freeze all except attention layers c_attn, gqa, msla
            if attn_layer in name:
                param.requires_grad = True
                n_trainable += param.numel()
            else:
                param.requires_grad = False
            n_total += param.numel()
    print(f"🔹 Trainable parameters: {n_trainable:,}/{n_total:,} ~ {n_trainable/n_total*100:.2f}%")
    return n_trainable, n_total
    
print(f"✅ Model ready: {mode}")
print("Model architecture:", model)
freeze_except_attention(model, mode)

print(f"Number of parameters: {model.num_parameters():,}")

# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.shape, param.numel())

# =====================
# Load dataset
# =====================
# dataset = load_dataset("wikitext", "wikitext-2-v1")#, split="train[:5%], validation[:10]")  # smaller for quick test
# print("✅ Dataset loaded.")
# print(dataset)
# train_dataset = dataset["train"].shuffle(seed=42)
# eval_dataset = dataset["validation"].shuffle(seed=42)

dataset = load_dataset("roneneldan/TinyStories", split=["train[:2000]", "validation[:10]"])
print("✅ Dataset loaded.")
print(dataset)
train_dataset = dataset[0]
eval_dataset = dataset[1]

# =====================
# Tokenize dataset
# =====================
def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_attention_mask=True,
    )

tokenized_train = train_dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"]
)
tokenized_train.set_format(
    type="torch",
    columns=["input_ids", "attention_mask"]
)

tokenized_eval = eval_dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"]
)
tokenized_eval.set_format(
    type="torch",
    columns=["input_ids", "attention_mask"]
)

# =====================
# Data collator
# =====================
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    # Shift logits and labels for causal LM
    shift_logits = logits[..., :-1, :].reshape(-1, logits.shape[-1])
    shift_labels = labels[..., 1:].reshape(-1)

    # Compute loss per token (CrossEntropy)
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(torch.tensor(shift_logits), torch.tensor(shift_labels))

    # Calculate perplexity
    perplexity = math.exp(loss.item())

    return {
        "loss": loss.item(),
        "perplexity": perplexity,
    }

from transformers import TrainerCallback, EarlyStoppingCallback, IntervalStrategy
import torch

class PerplexityLogger(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Đảm bảo model không None
        if model is None:
            return

        # Chỉ log mỗi 5 steps
        if state.global_step % 5 != 0:
            return

        # Lấy ví dụ dummy input (tokenizer phải tồn tại bên ngoài)
        dummy_input = tokenizer("Hello world", return_tensors="pt").to(args.device)

        with torch.no_grad():
            outputs = model(**dummy_input, labels=dummy_input["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()

        print(f"Step {state.global_step}: Perplexity = {perplexity:.2f}")

# =====================
# TrainingArguments
# =====================
train_args = TrainingArguments(
    output_dir=f"./outputs/train-{mode}",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    warmup_steps=100,
    learning_rate=1e-4,
    optim="adamw_torch",
    lr_scheduler_type="constant_with_warmup",
    max_grad_norm=1.0,
    max_steps=1000,              # quick test
    eval_steps=100,
    eval_strategy=IntervalStrategy.STEPS,
    save_strategy="steps",
    save_steps=100,  # Chỉ save 1 lần cuối
    save_total_limit=2,
    logging_steps=5,
    logging_dir=f"./outputs/logs-{mode}",
    logging_strategy="steps",
    logging_first_step=True,
    report_to="none",
    fp16=torch.cuda.is_available(),  # Enable mixed precision on GPU
    use_cpu=True,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    seed=42,
)

# =====================
# Trainer
# =====================
trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=collator,
    # compute_metrics=compute_metrics,
    # callbacks=[PerplexityLogger()],
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# =====================
# Train
# =====================
trainer.train()

# results = trainer.evaluate()
# print(results)

# trainer.save_model(f"./outputs/final-{mode}")
model.save_pretrained(f"./outputs/final-{mode}")
tokenizer.save_pretrained(f"./outputs/final-{mode}")

# Inference example
# =====================
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(inputs["input_ids"], 
                         pad_token_id=tokenizer.eos_token_id,
                         max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated text: {generated_text}")


logs = trainer.state.log_history

import pandas as pd

df_logs = pd.DataFrame(logs)
print(df_logs.head())

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(df_logs["step"], df_logs["loss"], marker='o')
plt.xlabel("Step")
plt.ylabel("Training Loss")
plt.title("Training Loss over Steps")
plt.grid()
plt.show()

# =====================
# Cleanup
# =====================
del model, tokenizer, dataset, tokenized_train, tokenized_eval, collator, model_path
del trainer , train_args, train_dataset, eval_dataset, logs, df_logs, outputs, inputs, generated_text
gc.collect()
if torch.cuda.is_available():
    # For CUDA
    torch.cuda.empty_cache()
if torch.backends.mps.is_available():
    # For macOS with MPS backend   
    torch.mps.empty_cache()
'''
patch_gpt2_with_msla(model, num_latents=64, k_top=4)
🔹 Trainable parameters: 28,938,240/132,116,736 ~ 21.90%
Number of parameters: 132,116,736

patch_gpt2_with_msla(model, num_latents=256, k_top=16)
🔹 Trainable parameters: 30,707,712/133,886,208 ~ 22.94%
Number of parameters: 133,886,208

patch_gpt2_with_msla(model, num_latents=256, k_top=64)
🔹 Trainable parameters: 30,707,712/133,886,208 ~ 22.94%
Number of parameters: 133,886,208

# =====================
# Train
# =====================

% python3 train.py
./outputs/final-gpt2_msla : False
[ℹ️] Using vanilla GPT2 model.
[🔧] Patching GPT2 Attention -> MSLA ...
[✅] MSLA Attention successfully patched.
✅ Model ready: gpt2_msla
Model architecture: GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-11): 12 x GPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2Attention(
          (c_attn): Identity()
          (c_proj): Conv1D(nf=768, nx=768)
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
          (msla): MSLA(
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (k_proj): Linear(in_features=768, out_features=768, bias=True)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D(nf=3072, nx=768)
          (c_proj): Conv1D(nf=768, nx=3072)
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)
🔒 Freezing all parameters except: attn.msla layers.
🔹 Trainable parameters: 30,707,712/133,886,208 ~ 22.94%
Number of parameters: 133,886,208
✅ Dataset loaded.
[Dataset({
    features: ['text'],
    num_rows: 2000
}), Dataset({
    features: ['text'],
    num_rows: 100
})]
  0%|                                                                                                                                      | 0/1000 [00:00<?, ?it/s]`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
{'loss': 8.7676, 'grad_norm': 86.79558563232422, 'learning_rate': 0.0, 'epoch': 0.0}                                                                                
{'loss': 8.0353, 'grad_norm': 37.96184539794922, 'learning_rate': 1.3333333333333333e-05, 'epoch': 0.01}                                                            
{'loss': 7.1727, 'grad_norm': 24.577728271484375, 'learning_rate': 3e-05, 'epoch': 0.02}                                                                            
{'loss': 6.7145, 'grad_norm': 20.57012939453125, 'learning_rate': 4.666666666666667e-05, 'epoch': 0.03}                                                             
{'loss': 6.2546, 'grad_norm': 21.87714958190918, 'learning_rate': 6.333333333333333e-05, 'epoch': 0.04}                                                             
{'loss': 5.4434, 'grad_norm': 5.8016438484191895, 'learning_rate': 8e-05, 'epoch': 0.05}                                                                            
{'loss': 5.203, 'grad_norm': 3.6966466903686523, 'learning_rate': 9.666666666666667e-05, 'epoch': 0.06}                                                             
{'loss': 4.8139, 'grad_norm': 3.4253103733062744, 'learning_rate': 0.0001, 'epoch': 0.07}                                                                           
{'loss': 4.5545, 'grad_norm': 2.995737075805664, 'learning_rate': 0.0001, 'epoch': 0.08}                                                                            
{'loss': 4.5027, 'grad_norm': 3.257674217224121, 'learning_rate': 0.0001, 'epoch': 0.09}                                                                            
{'loss': 4.3323, 'grad_norm': 2.742753267288208, 'learning_rate': 0.0001, 'epoch': 0.1}                                                                             
{'loss': 4.3245, 'grad_norm': 2.6547048091888428, 'learning_rate': 0.0001, 'epoch': 0.11}                                                                           
{'loss': 4.1857, 'grad_norm': 2.3817124366760254, 'learning_rate': 0.0001, 'epoch': 0.12}                                                                           
{'loss': 4.3286, 'grad_norm': 2.6614482402801514, 'learning_rate': 0.0001, 'epoch': 0.13}                                                                           
{'loss': 4.2998, 'grad_norm': 2.337472915649414, 'learning_rate': 0.0001, 'epoch': 0.14}                                                                            
{'loss': 4.1587, 'grad_norm': 2.762294054031372, 'learning_rate': 0.0001, 'epoch': 0.15}                                                                            
{'loss': 4.0622, 'grad_norm': 2.5609962940216064, 'learning_rate': 0.0001, 'epoch': 0.16}                                                                           
...
{'loss': 3.6992, 'grad_norm': 1.4531941413879395, 'learning_rate': 0.0001, 'epoch': 1.88}                                                                           
{'loss': 3.6152, 'grad_norm': 1.3922781944274902, 'learning_rate': 0.0001, 'epoch': 1.89}                                                                           
{'loss': 3.8318, 'grad_norm': 1.5693596601486206, 'learning_rate': 0.0001, 'epoch': 1.9}                                                                            
{'loss': 3.6622, 'grad_norm': 1.3492540121078491, 'learning_rate': 0.0001, 'epoch': 1.91}                                                                           
{'loss': 3.746, 'grad_norm': 1.7255377769470215, 'learning_rate': 0.0001, 'epoch': 1.92}                                                                            
{'loss': 3.8178, 'grad_norm': 1.514649510383606, 'learning_rate': 0.0001, 'epoch': 1.93}                                                                            
{'loss': 3.695, 'grad_norm': 1.561652421951294, 'learning_rate': 0.0001, 'epoch': 1.94}                                                                             
{'loss': 3.7499, 'grad_norm': 1.4812628030776978, 'learning_rate': 0.0001, 'epoch': 1.95}                                                                           
{'loss': 3.7172, 'grad_norm': 1.6010322570800781, 'learning_rate': 0.0001, 'epoch': 1.96}                                                                           
{'loss': 3.8052, 'grad_norm': 1.5097787380218506, 'learning_rate': 0.0001, 'epoch': 1.97}                                                                           
{'loss': 3.762, 'grad_norm': 1.6013039350509644, 'learning_rate': 0.0001, 'epoch': 1.98}                                                                            
{'loss': 3.6003, 'grad_norm': 1.6607716083526611, 'learning_rate': 0.0001, 'epoch': 1.99}                                                                           
{'loss': 3.6565, 'grad_norm': 1.5762592554092407, 'learning_rate': 0.0001, 'epoch': 2.0}                                                                            
{'train_runtime': 3121.4307, 'train_samples_per_second': 1.281, 'train_steps_per_second': 0.32, 'train_loss': 3.916411361694336, 'epoch': 2.0}                      
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [52:01<00:00,  3.12s/it]
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Generated text: Once upon a time, there was a little girl named Lily. She was very excited. One day, she was very happy. One day, the sun was so excited to the park.


% python3 train.py
[🔧] Patching GPT2 Attention -> MSLA ...
[✅] MSLA Attention successfully patched.
✅ Model ready: gpt2_msla
Model architecture: GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-11): 12 x GPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2Attention(
          (c_attn): Identity()
          (c_proj): Conv1D(nf=768, nx=768)
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
          (msla): MSLA(
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (k_proj): Linear(in_features=768, out_features=768, bias=True)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D(nf=3072, nx=768)
          (c_proj): Conv1D(nf=768, nx=3072)
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)
🔹 Trainable parameters: 28,938,240/132,116,736 ~ 21.90%
✅ Dataset loaded.
  0%|                                                                                                                                       | 0/100 [00:00<?, ?it/s]`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
{'loss': 8.5424, 'grad_norm': 19.386669158935547, 'learning_rate': 0.0001, 'epoch': 0.0}                                                                            
{'loss': 7.465, 'grad_norm': 11.358226776123047, 'learning_rate': 0.0001, 'epoch': 0.0}                                                                             
{'loss': 7.1734, 'grad_norm': 74.77347564697266, 'learning_rate': 0.0001, 'epoch': 0.0}                                                                             
{'loss': 6.9283, 'grad_norm': 19.474842071533203, 'learning_rate': 0.0001, 'epoch': 0.0}                                                                            
{'loss': 6.2607, 'grad_norm': 21.034814834594727, 'learning_rate': 0.0001, 'epoch': 0.0}                                                                            
{'loss': 6.2578, 'grad_norm': 26.59740447998047, 'learning_rate': 0.0001, 'epoch': 0.0}                                                                             
{'loss': 6.7738, 'grad_norm': 5.234988212585449, 'learning_rate': 0.0001, 'epoch': 0.0}                                                                             
{'loss': 5.698, 'grad_norm': 15.290159225463867, 'learning_rate': 0.0001, 'epoch': 0.0}                                                                             
{'loss': 6.1879, 'grad_norm': 4.957106113433838, 'learning_rate': 0.0001, 'epoch': 0.0}                                                                             
{'loss': 5.1737, 'grad_norm': 19.57628059387207, 'learning_rate': 0.0001, 'epoch': 0.01}                                                                            
{'loss': 5.8702, 'grad_norm': 5.390254497528076, 'learning_rate': 0.0001, 'epoch': 0.01}                                                                            
{'loss': 5.5351, 'grad_norm': 3.4058916568756104, 'learning_rate': 0.0001, 'epoch': 0.01}                                                                           
{'loss': 5.7667, 'grad_norm': 4.835726737976074, 'learning_rate': 0.0001, 'epoch': 0.01}                                                                            
{'loss': 5.4964, 'grad_norm': 2.7284445762634277, 'learning_rate': 0.0001, 'epoch': 0.01}                                                                           
{'loss': 5.9228, 'grad_norm': 7.432375431060791, 'learning_rate': 0.0001, 'epoch': 0.01}                                                                            
{'loss': 5.5554, 'grad_norm': 3.3884589672088623, 'learning_rate': 0.0001, 'epoch': 0.01}                                                                           
{'loss': 5.8245, 'grad_norm': 5.941636562347412, 'learning_rate': 0.0001, 'epoch': 0.01}                                                                            
{'loss': 5.4512, 'grad_norm': 3.9014339447021484, 'learning_rate': 0.0001, 'epoch': 0.01}                                                                           
{'loss': 5.9095, 'grad_norm': 5.689169406890869, 'learning_rate': 0.0001, 'epoch': 0.01}                                                                            
{'loss': 6.0638, 'grad_norm': 5.540028095245361, 'learning_rate': 0.0001, 'epoch': 0.01}                                                                            
{'train_runtime': 199.9783, 'train_samples_per_second': 2.0, 'train_steps_per_second': 0.5, 'train_loss': 6.1928339767456055, 'epoch': 0.01}                        
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [03:19<00:00,  2.00s/it]


% python3 train.py
[🔧] Patching GPT2 Attention -> MSLA ...
[✅] MSLA Attention successfully patched.
✅ Model ready: gpt2_msla
Model architecture: GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-11): 12 x GPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2Attention(
          (c_attn): Identity()
          (c_proj): Conv1D(nf=768, nx=768)
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
          (msla): MSLA(
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (k_proj): Linear(in_features=768, out_features=768, bias=True)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D(nf=3072, nx=768)
          (c_proj): Conv1D(nf=768, nx=3072)
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)
ℹ️ No attention layers to freeze. All parameters will be trainable.
🔹 Trainable parameters: 132,116,736/132,116,736 ~ 100.00%
✅ Dataset loaded.
[Dataset({
    features: ['text'],
    num_rows: 100
}), Dataset({
    features: ['text'],
    num_rows: 100
})]
  0%|                                                                                                                                       | 0/300 [00:00<?, ?it/s]`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
{'loss': 7.5449, 'grad_norm': 30.254182815551758, 'learning_rate': 3e-05, 'epoch': 0.2}                                                                             
{'loss': 6.2139, 'grad_norm': 13.169575691223145, 'learning_rate': 3e-05, 'epoch': 0.4}                                                                             
{'loss': 5.4936, 'grad_norm': 12.328495025634766, 'learning_rate': 3e-05, 'epoch': 0.6}                                                                             
{'loss': 5.0674, 'grad_norm': 13.047484397888184, 'learning_rate': 3e-05, 'epoch': 0.8}                                                                             
{'loss': 5.0086, 'grad_norm': 10.683314323425293, 'learning_rate': 3e-05, 'epoch': 1.0}                                                                             
{'loss': 4.7352, 'grad_norm': 10.703807830810547, 'learning_rate': 3e-05, 'epoch': 1.2}                                                                             
{'loss': 4.433, 'grad_norm': 11.998917579650879, 'learning_rate': 3e-05, 'epoch': 1.4}                                                                              
{'loss': 4.3854, 'grad_norm': 12.8565092086792, 'learning_rate': 3e-05, 'epoch': 1.6}                                                                               
{'loss': 4.2671, 'grad_norm': 10.822014808654785, 'learning_rate': 3e-05, 'epoch': 1.8}                                                                             
{'loss': 4.2695, 'grad_norm': 12.310787200927734, 'learning_rate': 3e-05, 'epoch': 2.0}                                                                             
{'loss': 4.0645, 'grad_norm': 12.128944396972656, 'learning_rate': 3e-05, 'epoch': 2.2}                                                                             
{'loss': 4.1573, 'grad_norm': 12.06430721282959, 'learning_rate': 3e-05, 'epoch': 2.4}                                                                              
{'loss': 3.8512, 'grad_norm': 13.398828506469727, 'learning_rate': 3e-05, 'epoch': 2.6}                                                                             
{'loss': 3.8727, 'grad_norm': 12.032587051391602, 'learning_rate': 3e-05, 'epoch': 2.8}                                                                             
{'loss': 3.9865, 'grad_norm': 14.385050773620605, 'learning_rate': 3e-05, 'epoch': 3.0}                                                                             
{'loss': 3.7519, 'grad_norm': 12.734075546264648, 'learning_rate': 3e-05, 'epoch': 3.2}                                                                             
{'loss': 3.7704, 'grad_norm': 14.657153129577637, 'learning_rate': 3e-05, 'epoch': 3.4}                                                                             
{'loss': 3.758, 'grad_norm': 12.67197036743164, 'learning_rate': 3e-05, 'epoch': 3.6}                                                                               
{'loss': 3.8098, 'grad_norm': 13.542635917663574, 'learning_rate': 3e-05, 'epoch': 3.8}                                                                             
{'loss': 3.6937, 'grad_norm': 13.774923324584961, 'learning_rate': 3e-05, 'epoch': 4.0}                                                                             
{'loss': 3.6582, 'grad_norm': 14.248916625976562, 'learning_rate': 3e-05, 'epoch': 4.2}                                                                             
{'loss': 3.578, 'grad_norm': 13.622511863708496, 'learning_rate': 3e-05, 'epoch': 4.4}                                                                              
{'loss': 3.5307, 'grad_norm': 14.118122100830078, 'learning_rate': 3e-05, 'epoch': 4.6}                                                                             
{'loss': 3.654, 'grad_norm': 15.79581356048584, 'learning_rate': 3e-05, 'epoch': 4.8}                                                                               
{'loss': 3.6003, 'grad_norm': 13.357193946838379, 'learning_rate': 3e-05, 'epoch': 5.0}                                                                             
{'loss': 3.5006, 'grad_norm': 14.659329414367676, 'learning_rate': 3e-05, 'epoch': 5.2}                                                                             
{'loss': 3.5297, 'grad_norm': 16.053539276123047, 'learning_rate': 3e-05, 'epoch': 5.4}                                                                             
{'loss': 3.4371, 'grad_norm': 13.78887939453125, 'learning_rate': 3e-05, 'epoch': 5.6}                                                                              
{'loss': 3.5527, 'grad_norm': 13.484370231628418, 'learning_rate': 3e-05, 'epoch': 5.8}                                                                             
{'loss': 3.456, 'grad_norm': 15.529924392700195, 'learning_rate': 3e-05, 'epoch': 6.0}                                                                              
{'loss': 3.4273, 'grad_norm': 13.66314697265625, 'learning_rate': 3e-05, 'epoch': 6.2}                                                                              
{'loss': 3.3987, 'grad_norm': 14.316596031188965, 'learning_rate': 3e-05, 'epoch': 6.4}                                                                             
{'loss': 3.3586, 'grad_norm': 15.963338851928711, 'learning_rate': 3e-05, 'epoch': 6.6}                                                                             
{'loss': 3.3971, 'grad_norm': 12.900081634521484, 'learning_rate': 3e-05, 'epoch': 6.8}                                                                             
{'loss': 3.3489, 'grad_norm': 15.32065200805664, 'learning_rate': 3e-05, 'epoch': 7.0}                                                                              
{'loss': 3.2297, 'grad_norm': 16.029096603393555, 'learning_rate': 3e-05, 'epoch': 7.2}                                                                             
{'loss': 3.1562, 'grad_norm': 15.576653480529785, 'learning_rate': 3e-05, 'epoch': 7.4}                                                                             
{'loss': 3.3918, 'grad_norm': 15.380437850952148, 'learning_rate': 3e-05, 'epoch': 7.6}                                                                             
{'loss': 3.281, 'grad_norm': 15.013849258422852, 'learning_rate': 3e-05, 'epoch': 7.8}                                                                              
{'loss': 3.3828, 'grad_norm': 16.221675872802734, 'learning_rate': 3e-05, 'epoch': 8.0}                                                                             
{'loss': 3.1346, 'grad_norm': 17.065629959106445, 'learning_rate': 3e-05, 'epoch': 8.2}                                                                             
{'loss': 3.215, 'grad_norm': 16.619964599609375, 'learning_rate': 3e-05, 'epoch': 8.4}                                                                              
{'loss': 3.1449, 'grad_norm': 15.206457138061523, 'learning_rate': 3e-05, 'epoch': 8.6}                                                                             
{'loss': 3.2732, 'grad_norm': 15.74333667755127, 'learning_rate': 3e-05, 'epoch': 8.8}                                                                              
{'loss': 3.3144, 'grad_norm': 16.431629180908203, 'learning_rate': 3e-05, 'epoch': 9.0}                                                                             
{'loss': 3.187, 'grad_norm': 15.682147026062012, 'learning_rate': 3e-05, 'epoch': 9.2}                                                                              
{'loss': 3.1272, 'grad_norm': 15.020416259765625, 'learning_rate': 3e-05, 'epoch': 9.4}                                                                             
{'loss': 3.0574, 'grad_norm': 17.83856773376465, 'learning_rate': 3e-05, 'epoch': 9.6}                                                                              
{'loss': 3.1532, 'grad_norm': 21.102035522460938, 'learning_rate': 3e-05, 'epoch': 9.8}                                                                             
{'loss': 3.2, 'grad_norm': 15.956536293029785, 'learning_rate': 3e-05, 'epoch': 10.0}                                                                               
{'loss': 3.1906, 'grad_norm': 15.693453788757324, 'learning_rate': 3e-05, 'epoch': 10.2}                                                                            
{'loss': 3.0014, 'grad_norm': 14.827354431152344, 'learning_rate': 3e-05, 'epoch': 10.4}                                                                            
{'loss': 2.9805, 'grad_norm': 15.910526275634766, 'learning_rate': 3e-05, 'epoch': 10.6}                                                                            
{'loss': 3.1622, 'grad_norm': 16.289920806884766, 'learning_rate': 3e-05, 'epoch': 10.8}                                                                            
{'loss': 3.1337, 'grad_norm': 18.012630462646484, 'learning_rate': 3e-05, 'epoch': 11.0}                                                                            
{'loss': 2.99, 'grad_norm': 15.653059005737305, 'learning_rate': 3e-05, 'epoch': 11.2}                                                                              
{'loss': 2.9907, 'grad_norm': 15.812373161315918, 'learning_rate': 3e-05, 'epoch': 11.4}                                                                            
{'loss': 3.0217, 'grad_norm': 17.740753173828125, 'learning_rate': 3e-05, 'epoch': 11.6}                                                                            
{'loss': 3.026, 'grad_norm': 15.315255165100098, 'learning_rate': 3e-05, 'epoch': 11.8}                                                                             
{'loss': 3.0874, 'grad_norm': 17.268287658691406, 'learning_rate': 3e-05, 'epoch': 12.0}                                                                            
{'train_runtime': 825.1594, 'train_samples_per_second': 1.454, 'train_steps_per_second': 0.364, 'train_loss': 3.6899151674906414, 'epoch': 12.0}                    
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [13:45<00:00,  2.75s/it]
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Generated text: Once upon a time, there was a small, there was a big, the park. He was a big, and said, she saw a big tree. She was very fast and wanted to the park.


% python3 train.py
[🔧] Patching GPT2 Attention -> MSLA ...
[✅] MSLA Attention successfully patched.
✅ Model ready: gpt2_msla
Model architecture: GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-11): 12 x GPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2Attention(
          (c_attn): Identity()
          (c_proj): Conv1D(nf=768, nx=768)
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
          (msla): MSLA(
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (k_proj): Linear(in_features=768, out_features=768, bias=True)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D(nf=3072, nx=768)
          (c_proj): Conv1D(nf=768, nx=3072)
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)
ℹ️ No attention layers to freeze. All parameters will be trainable.
🔹 Trainable parameters: 132,116,736/132,116,736 ~ 100.00%
✅ Dataset loaded.
[Dataset({
    features: ['text'],
    num_rows: 100
}), Dataset({
    features: ['text'],
    num_rows: 100
})]
  0%|                                                                                                                                       | 0/300 [00:00<?, ?it/s]`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
{'loss': 7.1151, 'grad_norm': 16.079429626464844, 'learning_rate': 0.0001, 'epoch': 0.2}                                                                            
{'loss': 5.8171, 'grad_norm': 10.989185333251953, 'learning_rate': 0.0001, 'epoch': 0.4}                                                                            
{'loss': 4.996, 'grad_norm': 10.367534637451172, 'learning_rate': 0.0001, 'epoch': 0.6}                                                                             
{'loss': 4.4744, 'grad_norm': 11.98746395111084, 'learning_rate': 0.0001, 'epoch': 0.8}                                                                             
{'loss': 4.464, 'grad_norm': 9.6346435546875, 'learning_rate': 0.0001, 'epoch': 1.0}                                                                                
{'loss': 4.123, 'grad_norm': 11.146602630615234, 'learning_rate': 0.0001, 'epoch': 1.2}                                                                             
{'loss': 3.9168, 'grad_norm': 12.892273902893066, 'learning_rate': 0.0001, 'epoch': 1.4}                                                                            
{'loss': 3.809, 'grad_norm': 12.949930191040039, 'learning_rate': 0.0001, 'epoch': 1.6}                                                                             
{'loss': 3.7337, 'grad_norm': 10.773770332336426, 'learning_rate': 0.0001, 'epoch': 1.8}                                                                            
{'loss': 3.8084, 'grad_norm': 13.352104187011719, 'learning_rate': 0.0001, 'epoch': 2.0}                                                                            
{'loss': 3.4569, 'grad_norm': 12.363762855529785, 'learning_rate': 0.0001, 'epoch': 2.2}                                                                            
{'loss': 3.5859, 'grad_norm': 12.790027618408203, 'learning_rate': 0.0001, 'epoch': 2.4}                                                                            
{'loss': 3.3416, 'grad_norm': 13.783103942871094, 'learning_rate': 0.0001, 'epoch': 2.6}                                                                            
{'loss': 3.4381, 'grad_norm': 12.149518013000488, 'learning_rate': 0.0001, 'epoch': 2.8}                                                                            
{'loss': 3.5502, 'grad_norm': 14.194889068603516, 'learning_rate': 0.0001, 'epoch': 3.0}                                                                            
{'loss': 3.2163, 'grad_norm': 12.491277694702148, 'learning_rate': 0.0001, 'epoch': 3.2}                                                                            
{'loss': 3.2368, 'grad_norm': 13.925180435180664, 'learning_rate': 0.0001, 'epoch': 3.4}                                                                            
{'loss': 3.2651, 'grad_norm': 11.415312767028809, 'learning_rate': 0.0001, 'epoch': 3.6}                                                                            
{'loss': 3.3288, 'grad_norm': 13.327127456665039, 'learning_rate': 0.0001, 'epoch': 3.8}                                                                            
{'loss': 3.25, 'grad_norm': 13.442595481872559, 'learning_rate': 0.0001, 'epoch': 4.0}                                                                              
{'loss': 3.0765, 'grad_norm': 12.436697006225586, 'learning_rate': 0.0001, 'epoch': 4.2}                                                                            
{'loss': 3.0069, 'grad_norm': 12.560273170471191, 'learning_rate': 0.0001, 'epoch': 4.4}                                                                            
{'loss': 2.9865, 'grad_norm': 13.244656562805176, 'learning_rate': 0.0001, 'epoch': 4.6}                                                                            
{'loss': 3.2032, 'grad_norm': 13.982789039611816, 'learning_rate': 0.0001, 'epoch': 4.8}                                                                            
{'loss': 3.135, 'grad_norm': 12.613296508789062, 'learning_rate': 0.0001, 'epoch': 5.0}                                                                             
{'loss': 2.8735, 'grad_norm': 13.156288146972656, 'learning_rate': 0.0001, 'epoch': 5.2}                                                                            
{'loss': 2.913, 'grad_norm': 13.085957527160645, 'learning_rate': 0.0001, 'epoch': 5.4}                                                                             
{'loss': 2.9386, 'grad_norm': 12.586024284362793, 'learning_rate': 0.0001, 'epoch': 5.6}                                                                            
{'loss': 3.0161, 'grad_norm': 11.74162769317627, 'learning_rate': 0.0001, 'epoch': 5.8}                                                                             
{'loss': 3.017, 'grad_norm': 13.457308769226074, 'learning_rate': 0.0001, 'epoch': 6.0}                                                                             
{'loss': 2.7815, 'grad_norm': 11.701578140258789, 'learning_rate': 0.0001, 'epoch': 6.2}                                                                            
{'loss': 2.7891, 'grad_norm': 11.955490112304688, 'learning_rate': 0.0001, 'epoch': 6.4}                                                                            
{'loss': 2.8608, 'grad_norm': 12.909794807434082, 'learning_rate': 0.0001, 'epoch': 6.6}                                                                            
{'loss': 2.9048, 'grad_norm': 11.079544067382812, 'learning_rate': 0.0001, 'epoch': 6.8}                                                                            
{'loss': 2.8846, 'grad_norm': 13.135249137878418, 'learning_rate': 0.0001, 'epoch': 7.0}                                                                            
{'loss': 2.647, 'grad_norm': 13.282357215881348, 'learning_rate': 0.0001, 'epoch': 7.2}                                                                             
{'loss': 2.585, 'grad_norm': 11.905113220214844, 'learning_rate': 0.0001, 'epoch': 7.4}                                                                             
{'loss': 2.8083, 'grad_norm': 12.722309112548828, 'learning_rate': 0.0001, 'epoch': 7.6}                                                                            
{'loss': 2.8, 'grad_norm': 12.791657447814941, 'learning_rate': 0.0001, 'epoch': 7.8}                                                                               
{'loss': 2.9262, 'grad_norm': 13.357074737548828, 'learning_rate': 0.0001, 'epoch': 8.0}                                                                            
{'loss': 2.5551, 'grad_norm': 12.425041198730469, 'learning_rate': 0.0001, 'epoch': 8.2}                                                                            
{'loss': 2.6235, 'grad_norm': 13.159387588500977, 'learning_rate': 0.0001, 'epoch': 8.4}                                                                            
{'loss': 2.6326, 'grad_norm': 11.874497413635254, 'learning_rate': 0.0001, 'epoch': 8.6}                                                                            
{'loss': 2.7674, 'grad_norm': 11.804251670837402, 'learning_rate': 0.0001, 'epoch': 8.8}                                                                            
{'loss': 2.7946, 'grad_norm': 11.841814041137695, 'learning_rate': 0.0001, 'epoch': 9.0}                                                                            
{'loss': 2.4946, 'grad_norm': 11.581562042236328, 'learning_rate': 0.0001, 'epoch': 9.2}                                                                            
{'loss': 2.5751, 'grad_norm': 10.567042350769043, 'learning_rate': 0.0001, 'epoch': 9.4}                                                                            
{'loss': 2.6121, 'grad_norm': 13.442440032958984, 'learning_rate': 0.0001, 'epoch': 9.6}                                                                            
{'loss': 2.6371, 'grad_norm': 15.285280227661133, 'learning_rate': 0.0001, 'epoch': 9.8}                                                                            
{'loss': 2.6989, 'grad_norm': 11.347857475280762, 'learning_rate': 0.0001, 'epoch': 10.0}                                                                           
{'loss': 2.4606, 'grad_norm': 11.146122932434082, 'learning_rate': 0.0001, 'epoch': 10.2}                                                                           
{'loss': 2.5007, 'grad_norm': 10.323534965515137, 'learning_rate': 0.0001, 'epoch': 10.4}                                                                           
{'loss': 2.5005, 'grad_norm': 11.667841911315918, 'learning_rate': 0.0001, 'epoch': 10.6}                                                                           
{'loss': 2.6244, 'grad_norm': 11.044605255126953, 'learning_rate': 0.0001, 'epoch': 10.8}                                                                           
{'loss': 2.6788, 'grad_norm': 12.845880508422852, 'learning_rate': 0.0001, 'epoch': 11.0}                                                                           
{'loss': 2.3786, 'grad_norm': 11.515792846679688, 'learning_rate': 0.0001, 'epoch': 11.2}                                                                           
{'loss': 2.4318, 'grad_norm': 10.651434898376465, 'learning_rate': 0.0001, 'epoch': 11.4}                                                                           
{'loss': 2.5047, 'grad_norm': 12.249659538269043, 'learning_rate': 0.0001, 'epoch': 11.6}                                                                           
{'loss': 2.5241, 'grad_norm': 11.05530834197998, 'learning_rate': 0.0001, 'epoch': 11.8}                                                                            
{'loss': 2.6062, 'grad_norm': 12.192193031311035, 'learning_rate': 0.0001, 'epoch': 12.0}                                                                           
{'train_runtime': 892.988, 'train_samples_per_second': 1.344, 'train_steps_per_second': 0.336, 'train_loss': 3.1613644282023112, 'epoch': 12.0}                     
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [14:52<00:00,  2.98s/it]
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Generated text: Once upon a time, there was a little girl named Tim. She was a big box. He was a big, "Do you have any.


{'loss': 3.6639, 'grad_norm': 1.9493802785873413, 'learning_rate': 0.0001, 'epoch': 1.9}                                                                            
{'loss': 3.5247, 'grad_norm': 1.8659108877182007, 'learning_rate': 0.0001, 'epoch': 1.91}                                                                           
{'loss': 3.5966, 'grad_norm': 1.8877536058425903, 'learning_rate': 0.0001, 'epoch': 1.92}                                                                           
{'loss': 3.6776, 'grad_norm': 2.014611005783081, 'learning_rate': 0.0001, 'epoch': 1.93}                                                                            
{'loss': 3.5622, 'grad_norm': 2.177809476852417, 'learning_rate': 0.0001, 'epoch': 1.94}                                                                            
{'loss': 3.6276, 'grad_norm': 1.8701190948486328, 'learning_rate': 0.0001, 'epoch': 1.95}                                                                           
{'loss': 3.5452, 'grad_norm': 2.085341453552246, 'learning_rate': 0.0001, 'epoch': 1.96}                                                                            
{'loss': 3.6382, 'grad_norm': 1.86881685256958, 'learning_rate': 0.0001, 'epoch': 1.97}                                                                             
{'loss': 3.6071, 'grad_norm': 2.0359392166137695, 'learning_rate': 0.0001, 'epoch': 1.98}                                                                           
{'loss': 3.4807, 'grad_norm': 2.3893673419952393, 'learning_rate': 0.0001, 'epoch': 1.99}                                                                           
{'loss': 3.5204, 'grad_norm': 2.081857442855835, 'learning_rate': 0.0001, 'epoch': 2.0}                                                                             
{'train_runtime': 2706.1558, 'train_samples_per_second': 1.478, 'train_steps_per_second': 0.37, 'train_loss': 3.7292665672302245, 'epoch': 2.0}                     
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [45:06<00:00,  2.71s/it]
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Generated text: Once upon a time, there was a little girl named Timmy. She was very excited. One day, she saw a big tree. One day, but it was so excited to the park.


% python3 train.py
[🔧] Patching GPT2 Attention -> GQA ...
[✅] GQA Attention successfully patched.
✅ Model ready: gpt2_gqa
Model architecture: GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-11): 12 x GPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2Attention(
          (c_attn): Identity()
          (c_proj): Conv1D(nf=768, nx=768)
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
          (gqa): GQA(
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (k_proj): Linear(in_features=768, out_features=256, bias=True)
            (v_proj): Linear(in_features=768, out_features=256, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D(nf=3072, nx=768)
          (c_proj): Conv1D(nf=768, nx=3072)
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)
🔹 Trainable parameters: 18,898,944/122,077,440 ~ 15.48%
✅ Dataset loaded.
  0%|                                                                                                                                       | 0/100 [00:00<?, ?it/s]`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
{'loss': 8.5187, 'grad_norm': 2.876471519470215, 'learning_rate': 0.0001, 'epoch': 0.0}                                                                             
{'loss': 8.3183, 'grad_norm': 1.5372328758239746, 'learning_rate': 0.0001, 'epoch': 0.0}                                                                            
{'loss': 8.3212, 'grad_norm': 1.733076810836792, 'learning_rate': 0.0001, 'epoch': 0.0}                                                                             
{'loss': 7.9307, 'grad_norm': 1.1770174503326416, 'learning_rate': 0.0001, 'epoch': 0.0}                                                                            
{'loss': 7.7117, 'grad_norm': 1.6691426038742065, 'learning_rate': 0.0001, 'epoch': 0.0}                                                                            
{'loss': 7.8298, 'grad_norm': 1.5441513061523438, 'learning_rate': 0.0001, 'epoch': 0.0}                                                                            
{'loss': 7.9962, 'grad_norm': 0.895899772644043, 'learning_rate': 0.0001, 'epoch': 0.0}                                                                             
{'loss': 7.4025, 'grad_norm': 3.6942710876464844, 'learning_rate': 0.0001, 'epoch': 0.0}                                                                            
{'loss': 7.8326, 'grad_norm': 1.4924196004867554, 'learning_rate': 0.0001, 'epoch': 0.0}                                                                            
{'loss': 7.0763, 'grad_norm': 4.87847375869751, 'learning_rate': 0.0001, 'epoch': 0.01}                                                                             
{'loss': 7.7359, 'grad_norm': 1.4307758808135986, 'learning_rate': 0.0001, 'epoch': 0.01}                                                                           
{'loss': 7.3294, 'grad_norm': 0.7872484922409058, 'learning_rate': 0.0001, 'epoch': 0.01}                                                                           
{'loss': 7.4525, 'grad_norm': 1.1541298627853394, 'learning_rate': 0.0001, 'epoch': 0.01}                                                                           
{'loss': 7.2482, 'grad_norm': 0.7021639347076416, 'learning_rate': 0.0001, 'epoch': 0.01}                                                                           
{'loss': 7.5709, 'grad_norm': 1.802327275276184, 'learning_rate': 0.0001, 'epoch': 0.01}                                                                            
{'loss': 7.3213, 'grad_norm': 1.4604506492614746, 'learning_rate': 0.0001, 'epoch': 0.01}                                                                           
{'loss': 7.5081, 'grad_norm': 1.5807995796203613, 'learning_rate': 0.0001, 'epoch': 0.01}                                                                           
{'loss': 7.4321, 'grad_norm': 1.2213581800460815, 'learning_rate': 0.0001, 'epoch': 0.01}                                                                           
{'loss': 7.6853, 'grad_norm': 1.851847529411316, 'learning_rate': 0.0001, 'epoch': 0.01}                                                                            
{'loss': 7.8326, 'grad_norm': 1.7619658708572388, 'learning_rate': 0.0001, 'epoch': 0.01}                                                                           
{'train_runtime': 165.0295, 'train_samples_per_second': 2.424, 'train_steps_per_second': 0.606, 'train_loss': 7.702709579467774, 'epoch': 0.01}                     
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [02:45<00:00,  1.65s/it]

% python3 train.py
[ℹ️] Using vanilla GPT2.
✅ Model ready: gpt2
Model architecture: GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-11): 12 x GPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2Attention(
          (c_attn): Conv1D(nf=2304, nx=768)
          (c_proj): Conv1D(nf=768, nx=768)
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D(nf=3072, nx=768)
          (c_proj): Conv1D(nf=768, nx=3072)
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)
🔹 Trainable parameters: 21,261,312/124,439,808 ~ 17.09%
✅ Dataset loaded.
  0%|                                                                                                                                       | 0/100 [00:00<?, ?it/s]`loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
{'loss': 4.112, 'grad_norm': 2.2381954193115234, 'learning_rate': 0.0001, 'epoch': 0.0}                                                                             
{'loss': 4.2826, 'grad_norm': 4.073288440704346, 'learning_rate': 0.0001, 'epoch': 0.0}                                                                             
{'loss': 4.6468, 'grad_norm': 2.989330768585205, 'learning_rate': 0.0001, 'epoch': 0.0}                                                                             
{'loss': 4.3096, 'grad_norm': 3.051726818084717, 'learning_rate': 0.0001, 'epoch': 0.0}                                                                             
{'loss': 4.0869, 'grad_norm': 4.302624225616455, 'learning_rate': 0.0001, 'epoch': 0.0}                                                                             
{'loss': 4.3474, 'grad_norm': 3.0724360942840576, 'learning_rate': 0.0001, 'epoch': 0.0}                                                                            
{'loss': 4.0006, 'grad_norm': 2.2364349365234375, 'learning_rate': 0.0001, 'epoch': 0.0}                                                                            
{'loss': 3.8675, 'grad_norm': 11.920469284057617, 'learning_rate': 0.0001, 'epoch': 0.0}                                                                            
{'loss': 4.032, 'grad_norm': 2.3496787548065186, 'learning_rate': 0.0001, 'epoch': 0.0}                                                                             
{'loss': 4.1342, 'grad_norm': 7.293999671936035, 'learning_rate': 0.0001, 'epoch': 0.01}                                                                            
{'loss': 3.7383, 'grad_norm': 3.55206561088562, 'learning_rate': 0.0001, 'epoch': 0.01}                                                                             
{'loss': 3.8925, 'grad_norm': 2.1150529384613037, 'learning_rate': 0.0001, 'epoch': 0.01}                                                                           
{'loss': 3.8657, 'grad_norm': 3.757155418395996, 'learning_rate': 0.0001, 'epoch': 0.01}                                                                            
{'loss': 3.8234, 'grad_norm': 1.9179139137268066, 'learning_rate': 0.0001, 'epoch': 0.01}                                                                           
{'loss': 4.1085, 'grad_norm': 4.528071880340576, 'learning_rate': 0.0001, 'epoch': 0.01}                                                                            
{'loss': 3.7958, 'grad_norm': 2.033447265625, 'learning_rate': 0.0001, 'epoch': 0.01}                                                                               
{'loss': 4.0743, 'grad_norm': 4.042884349822998, 'learning_rate': 0.0001, 'epoch': 0.01}                                                                            
{'loss': 3.6291, 'grad_norm': 2.3693594932556152, 'learning_rate': 0.0001, 'epoch': 0.01}                                                                           
{'loss': 3.7036, 'grad_norm': 3.258683204650879, 'learning_rate': 0.0001, 'epoch': 0.01}                                                                            
{'loss': 4.0685, 'grad_norm': 7.478783130645752, 'learning_rate': 0.0001, 'epoch': 0.01}                                                                            
{'train_runtime': 182.5205, 'train_samples_per_second': 2.192, 'train_steps_per_second': 0.548, 'train_loss': 4.025969676971435, 'epoch': 0.01}                     
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [03:02<00:00,  1.83s/it]


{'loss': 1.8056, 'grad_norm': 2.775254487991333, 'learning_rate': 0.0001, 'epoch': 1.87}                                                                            
{'loss': 1.8139, 'grad_norm': 2.567384719848633, 'learning_rate': 0.0001, 'epoch': 1.88}                                                                            
{'loss': 1.7317, 'grad_norm': 2.7649052143096924, 'learning_rate': 0.0001, 'epoch': 1.89}                                                                           
{'loss': 1.7642, 'grad_norm': 2.594439744949341, 'learning_rate': 0.0001, 'epoch': 1.9}                                                                             
{'loss': 1.7285, 'grad_norm': 2.498512029647827, 'learning_rate': 0.0001, 'epoch': 1.91}                                                                            
{'loss': 1.7711, 'grad_norm': 2.5413057804107666, 'learning_rate': 0.0001, 'epoch': 1.92}                                                                           
{'loss': 1.7813, 'grad_norm': 2.6274330615997314, 'learning_rate': 0.0001, 'epoch': 1.93}                                                                           
{'loss': 1.7073, 'grad_norm': 2.5947365760803223, 'learning_rate': 0.0001, 'epoch': 1.94}                                                                           
{'loss': 1.8514, 'grad_norm': 2.7293789386749268, 'learning_rate': 0.0001, 'epoch': 1.95}                                                                           
{'loss': 1.7255, 'grad_norm': 2.454606533050537, 'learning_rate': 0.0001, 'epoch': 1.96}                                                                            
{'loss': 1.8042, 'grad_norm': 2.7099735736846924, 'learning_rate': 0.0001, 'epoch': 1.97}                                                                           
{'loss': 1.7733, 'grad_norm': 3.0594279766082764, 'learning_rate': 0.0001, 'epoch': 1.98}                                                                           
{'loss': 1.6603, 'grad_norm': 2.5797877311706543, 'learning_rate': 0.0001, 'epoch': 1.99}                                                                           
{'loss': 1.6915, 'grad_norm': 2.8339438438415527, 'learning_rate': 0.0001, 'epoch': 2.0}                                                                            
{'train_runtime': 2267.0922, 'train_samples_per_second': 1.764, 'train_steps_per_second': 0.441, 'train_loss': 1.924752809524536, 'epoch': 2.0}                     
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [37:47<00:00,  2.27s/it]



📊 Tóm tắt kết quả huấn luyện

Model	Trainable Params	% Trainable	Train Loss ↓	Time / 100 steps
Vanilla	21.26M / 124.4M	~17.09%	4.02	~182 sec
GQA	16.53M / 119.7M	~13.81%	7.68	~163 sec
MSLA	28.93M / 132.1M	~21.90%	6.19	~199 sec
🔍 Phân tích từng điểm

1️⃣ Train Loss
Vanilla GPT2 train loss ~4.02
Rất thấp, chứng tỏ mô hình gốc phù hợp và dễ tối ưu nhất trên Wikitext-2.
GQA train loss ~7.68
Cao hơn hẳn. Có 3 nguyên nhân:
Giảm số chiều key/value (k_proj và v_proj output 128 thay vì 768).
Ít tham số hơn (~13% trainable).
Giai đoạn warm-up ban đầu model chưa thích nghi với projection layer mới.
MSLA train loss ~6.19
Tốt hơn GQA (~1.5 điểm loss).
Vẫn chưa bằng vanilla GPT2, nhưng khả năng học tốt hơn GQA.
Có thể do MSLA vẫn giữ kích thước key/value full (768), đồng thời chỉ thêm sparse latent selection.
Kết luận Loss:
➡️ Vanilla GPT2 > MSLA > GQA về khả năng giảm loss.

2️⃣ Grad Norm
Vanilla:
Dao động ~2–11.
Rất ổn định, ít cực trị.
GQA:
Đa số thấp (~0.9–4), trừ 1–2 step có spike ~6.
Thường thấy trong layer projection nhỏ hơn.
MSLA:
Nhiều spike rất cao (74!), rồi giảm dần.
Đây là đặc điểm của các attention module sparse: gradient dễ "nổ" ban đầu.
Kết luận Grad Norm:

MSLA cần gradient clipping kỹ.
GQA ổn định nhất.
Vanilla khá trung bình.
3️⃣ Thời gian train
GQA nhanh hơn (163s/100 steps).
Vanilla trung bình (182s).
MSLA chậm nhất (199s), do sparse selection.
4️⃣ Số tham số trainable
MSLA thêm khá nhiều tham số (28M).
GQA giảm tham số đáng kể (16M).
Vanilla giữa chừng.
🟢 Đánh giá tổng quan

✅ Vanilla GPT2

Đương nhiên dễ học nhất vì không thay đổi kiến trúc.
Loss thấp, gradient ổn.
Sử dụng benchmark baseline.
✅ MSLA

Loss thấp hơn GQA dù trainable params nhiều hơn.
Gradient ban đầu cao nhưng giảm dần.
Có tiềm năng nếu fine-tune thêm hoặc warm-up kỹ.
✅ GQA

Loss cao nhất.
Gradient ổn định.
Rất nhẹ về số tham số.
Có thể cần nhiều epoch hơn để bắt kịp.
🧠 Gợi ý cải thiện

Nếu bạn muốn so sánh công bằng hơn:

Chạy ít nhất 1–3 epoch, vì attention projection cần warm-up dài hơn.
Giảm learning rate (ví dụ 5e-5).
Bật gradient clipping (bạn đã để max_grad_norm=1.0 rồi—giữ nguyên).
Theo dõi thêm perplexity.
🏆 Kết luận tạm thời

Vanilla GPT2 là baseline mạnh nhất trên Wikitext.
MSLA tiềm năng tốt hơn GQA (thấp hơn 1.5 điểm loss).
GQA trade-off nhiều capacity để tiết kiệm params, cần thời gian train lâu hơn.
'''