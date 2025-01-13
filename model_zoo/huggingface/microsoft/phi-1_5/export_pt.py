import os
import torch
from transformers import PhiForCausalLM, AutoTokenizer

model_dir = '/project/ai/zhouyi_compass/Model_Resource/huggingface/microsoft/phi-1_5'
TORCH_DIR = os.path.join(model_dir, 'OUTPUTS/TORCH')
os.makedirs(TORCH_DIR, mode=0o775, exist_ok=True)

device = 'cpu'
torch.set_default_device(device)

model = PhiForCausalLM.from_pretrained(model_dir, **{'return_dict': False}).eval()
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

prompt = 'what is the first amendment of the Constitution?'
tokens = tokenizer(prompt, return_tensors='pt')

generated_output = model.generate(**tokens, max_length=256)
print(tokenizer.batch_decode(generated_output)[0])

traced_model = torch.jit.trace(model, [tokens['input_ids'], tokens['attention_mask']])
traced_model.save(os.path.join(TORCH_DIR, 'phi_1_5.pt'))
loaded_model = torch.jit.load(os.path.join(TORCH_DIR, 'phi_1_5.pt'), map_location=device, _restore_shapes=True)
loaded_output = loaded_model(tokens['input_ids'], tokens['attention_mask'])
pass
