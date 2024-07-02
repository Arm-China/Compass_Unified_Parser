import os
import torch
from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer


model_dir = '/project/ai/zhouyi_compass/Model_Resource/huggingface/state-spaces/mamba-130m-hf'
TORCH_DIR = os.path.join(model_dir, 'OUTPUTS/TORCH')
ONNX_DIR = os.path.join(model_dir, 'OUTPUTS/ONNX')
os.makedirs(TORCH_DIR, mode=0o775, exist_ok=True)
os.makedirs(ONNX_DIR, mode=0o775, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = MambaForCausalLM.from_pretrained(model_dir, **{'return_dict': False, 'use_cache': False})
model.eval()
prompt = 'Congress shall make no law respecting an establishment of religion, or prohibiting the free exercise thereof; or abridging the freedom of speech, or of the press, or the right of the people peaceably to assemble, and to petition the Government for a redress of grievances.'  # nopep8
input_ids = tokenizer(prompt, return_tensors='pt')['input_ids']

out = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.batch_decode(out))

pt_path = os.path.join(TORCH_DIR, 'mamba_130m_hf.pt')
onnx_path = os.path.join(ONNX_DIR, 'mamba_130m_hf.onnx')
traced_model = torch.jit.trace(model, input_ids)
traced_model.save(pt_path)
torch.onnx.export(traced_model, input_ids, onnx_path, verbose=True, do_constant_folding=False)
