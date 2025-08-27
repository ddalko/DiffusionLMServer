import torch

from generate import generate
from transformers import AutoTokenizer, AutoModel


def chat():
    device = 'cuda'
    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    gen_length = 128
    steps = 128
    print('*' * 66)
    print(f'**  Answer Length: {gen_length}  |  Sampling Steps: {steps}  **')
    print('*' * 66)

    m = [{"role": "user", "content": user_input}]
    user_input = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(user_input)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    prompt = input_ids
    out = generate(model, prompt, steps=steps, gen_length=gen_length, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    answer = tokenizer.batch_decode(out[:, prompt.shape[1]:], skip_special_tokens=True)[0]


if __name__ == "__main__":
    chat()
