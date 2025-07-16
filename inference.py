from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Đường dẫn đến checkpoint
checkpoint_path = "outputs/final-gpt2_msla-256x16_lr1e4_1000steps_only"  # chỉnh lại số checkpoint cuối cùng bạn có

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model
# model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
# model.eval()

# Load model với MSLA
# from gpt2_with_MSLA import patch_gpt2_with_msla, GPT2WithMSLA
# model = GPT2WithMSLA.from_pretrained_with_msla("outputs/final-gpt2_msla")


from gpt2_with_MSLA import from_pretrained_with_msla

model = from_pretrained_with_msla(
    checkpoint_path,
    num_latents=256,
    k_top=16,
    device="cpu"
)

# Load model từ checkpoint, KHÔNG patch trước
# model = GPT2LMHeadModel.from_pretrained("outputs/final-gpt2_msla")

# Patch sau khi load
# from gpt2_with_MSLA import patch_gpt2_with_msla
# patch_gpt2_with_msla(model)
model.eval()

# Nếu có GPU, dùng GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Input prompt
# prompt = "Once upon a time"
prompt = "The capital of France is"

def generate_text(prompt, max_length=120):
    # Encode input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    # Generate
    with torch.no_grad():
        output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_length,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        do_sample=True,
    )
    return output

for i in range(3):
    print("=" * 20)
    print(f"Generating text {i+1}...")
    # Generate text
    output = generate_text(prompt)

    # Decode output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Generated text:\n", generated_text)

'''
% python inference.py
[🔧] Loading GPT2 config...
[🔧] Instantiating GPT2 model...
[💾] Loading base state_dict...
[🔧] Loading base weights...
[✅] Base weights loaded. Missing: 25 keys, Unexpected: 108 keys
[🔧] Patching MSLA into GPT2...
[✅] All MSLA blocks patched and weights loaded.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Generated text:
 Once upon a time, there was a little girl named Max and he saw a big red and said, he ran up the forest. So he found a little girl and said, "Do you can fix the park to relax. He went to find something strange. It was very proud of her friends. She looked at the park to the animals to the birdcage was so she said, "Are you and he went back to help others. It was so excited! It looked at the barber's so surprised
'''