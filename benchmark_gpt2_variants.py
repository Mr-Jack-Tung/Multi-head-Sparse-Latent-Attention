# benchmark_gpt2_variants.py

import time
import torch
import math
import gc
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Load dataset
# dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
dataset = load_dataset("wikitext", "wikitext-2-v1")#, split="train[:5%], validation[:10]")  # smaller for quick test
print("✅ Dataset loaded.")

train_dataset = dataset["train"].shuffle(seed=42)
eval_dataset = dataset["validation"].shuffle(seed=42)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids"])

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

args = TrainingArguments(
    output_dir="./benchmark",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    # per_device_eval_batch_size=2,
    max_steps=100,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=3,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    logging_steps=5,
    save_strategy="no",
    report_to="none",
)

# Training + Evaluation helper
def benchmark_model(model, name="Model"):
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        # eval_dataset=dataset["validation"],
        # tokenizer=tokenizer,
        data_collator=collator,
    )

    print(f"\n🏃‍♂️ Training {name}...")
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    trainer.train()
    end = time.time()
    train_time = end - start
    max_memory = torch.cuda.max_memory_allocated() / 1e9

    eval_results = trainer.evaluate()
    ppl = math.exp(eval_results["eval_loss"])

    return {
        "train_time_sec": train_time,
        "loss": eval_results["eval_loss"],
        "perplexity": ppl,
        "max_memory_gb": max_memory
    }

# Inference helper
def benchmark_inference(model, name="Model"):
    print(f"\n⚡ Inference Benchmark {name}...")
    model.eval()
    torch.cuda.empty_cache()
    gc.collect()

    prompt = "The meaning of life is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    output = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False
    )
    torch.cuda.synchronize()
    end = time.time()
    duration = end - start
    tokens_generated = output.shape[1] - inputs["input_ids"].shape[1]
    tokens_per_sec = tokens_generated / duration

    return {
        "tokens_generated": tokens_generated,
        "duration_sec": duration,
        "tokens_per_sec": tokens_per_sec
    }

# Load GPT2
torch.cuda.empty_cache()
gc.collect()

model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
train_gpt2 = benchmark_model(model_gpt2, "GPT2 Original")
infer_gpt2 = benchmark_inference(model_gpt2, "GPT2 Original")

# Load GPT2 + MSLA
torch.cuda.empty_cache()
gc.collect()

from gpt2_with_MSLA import patch_gpt2_with_msla
model_msla = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
patch_gpt2_with_msla(model_msla, num_latents=64, k_top=4)
train_msla = benchmark_model(model_msla, "GPT2 + MSLA")
infer_msla = benchmark_inference(model_msla, "GPT2 + MSLA")

# Load GPT2 + GQA
torch.cuda.empty_cache()
gc.collect()

from gpt2_with_GQA import patch_gpt2_with_gqa
model_gqa = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
patch_gpt2_with_gqa(model_gqa, num_key_value_groups=2)
train_gqa = benchmark_model(model_gqa, "GPT2 + GQA")
infer_gqa = benchmark_inference(model_gqa, "GPT2 + GQA")

# Summary
print("\n📊 Benchmark Results:")
print("------------------------------------------------------------------")
print("Model        | TrainTime(s) | Perplexity | Loss   | Mem(GB) | TPS")
print("------------------------------------------------------------------")
print(f"GPT2 Original| {train_gpt2['train_time_sec']:.1f}        | {train_gpt2['perplexity']:.2f}      | {train_gpt2['loss']:.4f} | {train_gpt2['max_memory_gb']:.2f} | {infer_gpt2['tokens_per_sec']:.1f}")
print(f"GPT2 + MSLA  | {train_msla['train_time_sec']:.1f}        | {train_msla['perplexity']:.2f}      | {train_msla['loss']:.4f} | {train_msla['max_memory_gb']:.2f} | {infer_msla['tokens_per_sec']:.1f}")
print(f"GPT2 + GQA   | {train_gqa['train_time_sec']:.1f}        | {train_gqa['perplexity']:.2f}      | {train_gqa['loss']:.4f} | {train_gqa['max_memory_gb']:.2f} | {infer_gqa['tokens_per_sec']:.1f}")
