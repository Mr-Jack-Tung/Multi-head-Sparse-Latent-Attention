# Multi-head Sparse Latent Attention (MSLA) Project

This project implements and explores the Multi-head Sparse Latent Attention (MSLA) mechanism, a novel approach to attention in transformer models. MSLA aims to improve efficiency and performance by selectively focusing on relevant parts of the input sequence.

## Project Description

The MSLA project provides a framework for training and evaluating transformer models enhanced with sparse attention mechanisms. It includes implementations for training, inference, and benchmarking, allowing for a comprehensive analysis of MSLA's effectiveness compared to traditional attention methods.

## Usage Guide

This guide outlines how to use the provided scripts for training, inference, and benchmarking.

## Type of attention matrix

![alt-text](./Type%20of%20attention%20matrix.jpg)

### 1. Training

The `train.py` script is used for training models. You can train models with different configurations and attention mechanisms.

**Example:**
```bash
python train.py --model_name gpt2 --attention_type msla --learning_rate 1e-4 --num_steps 1000
```

### 2. Inference

The `inference.py` script is used for generating text or performing other inference tasks with trained models.

**Example:**
```bash
python inference.py --model_path outputs/final-gpt2_msla_lr1e4_1000steps_full --prompt "The future of AI is"
```

### 3. Benchmarking

The `benchmark_gpt2_variants.py` script allows you to compare the performance of different GPT-2 variants, including those with MSLA.

**Example:**
```bash
python benchmark_gpt2_variants.py
```

### Model Variants

The project includes specific scripts for different model configurations:
- `gpt2_with_MSLA.py`: Demonstrates GPT-2 with the MSLA attention mechanism.
- `gpt2_with_GQA.py`: Demonstrates GPT-2 with Grouped-Query Attention (GQA) for comparison.

These scripts can be used as examples or starting points for custom training and experimentation.

## Project Structure

- `train.py`: Script for training models.
- `inference.py`: Script for performing inference.
- `benchmark_gpt2_variants.py`: Script for benchmarking different GPT-2 variants.
- `gpt2_with_MSLA.py`: Example script for GPT-2 with MSLA.
- `gpt2_with_GQA.py`: Example script for GPT-2 with GQA.
- `outputs/`: Directory containing trained model checkpoints and configurations.

## Model Archtecture
```
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
```


## Training Result with 1000 steps only
```
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
✅ Dataset loaded: roneneldan/TinyStories
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
```
