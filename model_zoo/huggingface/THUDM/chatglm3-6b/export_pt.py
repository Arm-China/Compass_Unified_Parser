import sys
import os
import torch
from transformers import AutoTokenizer, AutoModel
remote_dir = '/project/ai/zhouyi_compass/Model_Resource/huggingface/THUDM/chatglm3-6b'  # nopep8
sys.path.append(remote_dir)  # nopep8

from modeling_chatglm import ChatGLMModel, ChatGLMForConditionalGeneration  # nopep8
from configuration_chatglm import ChatGLMConfig  # nopep8

question = 'who is founding father of the United States?'
print(question)

tokenizer = AutoTokenizer.from_pretrained(remote_dir, trust_remote_code=True)
tokens = tokenizer.encode(question)
print(tokens)

config = {'return_dict': False}

model = ChatGLMForConditionalGeneration.from_pretrained(remote_dir, **config).half().cuda()
model = model.eval()
response, history = model.chat(tokenizer, question, history=[], **config)
print(response)


inputs = torch.tensor([tokens], dtype=torch.int64, device='cuda')
#outputs = model(inputs)

traced_model = torch.jit.trace(model, inputs)
traced_outputs = traced_model(inputs)

output_dir = os.path.join(remote_dir, 'OUTPUTS/TORCH')
os.makedirs(output_dir, mode=0o775, exist_ok=True)
pt_path = os.path.join(output_dir, 'chatglm3-6b.pt')
traced_model.save(pt_path)
