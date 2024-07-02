import os
import numpy as np
import torch
from transformers import AutoTokenizer, BigBirdForMaskedLM


def sim(x, y):
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    return np.dot(x.flatten(), y.flatten()) / (np.linalg.norm(x) * np.linalg.norm(y))


model_dir = '/project/ai/zhouyi_compass/Model_Resource/huggingface/google/bigbird_roberta_base'
TORCH_DIR = os.path.join(model_dir, 'OUTPUTS/TORCH')
os.makedirs(TORCH_DIR, mode=0o775, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = BigBirdForMaskedLM.from_pretrained(model_dir, **{'return_dict': False, 'attention_type': 'original_full'})
model = model.eval()

LONG_ARTICLE_TARGET = 'To properly understand political power and trace its origins, we must consider the state that all people are in naturally. That is a state of perfect freedom of acting and disposing of their own possessions and persons as they think fit within the bounds of the law of nature. People in this state do not have to ask permission to act or depend on the will of others to arrange matters on their behalf. The natural state is also one of equality in which all power and jurisdiction is reciprocal and no one has more than another. It is evident that all human beings—as creatures belonging to the same species and rank and born indiscriminately with all the same natural advantages and faculties—are equal amongst themselves. They have no relationship of subordination or subjection unless God (the lord and master of them all) had clearly set one person above another and conferred on him an undoubted right to dominion and sovereignty.'

# add mask_token
LONG_ARTICLE_TO_MASK = LONG_ARTICLE_TARGET.replace('freedom', '[MASK]')
inputs = tokenizer(LONG_ARTICLE_TO_MASK, return_tensors='pt')
# long article input
print(list(inputs['input_ids'].shape))


with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs[0]

# retrieve index of [MASK]
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
retrieved_word = tokenizer.decode(predicted_token_id)
print(retrieved_word)

pt_path = os.path.join(TORCH_DIR, 'bigbird-roberta-base.pt')
traced_model = torch.jit.trace(model, example_kwarg_inputs={
                               'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}, strict=False)
traced_model.save(pt_path)
reloaded_model = torch.jit.load(pt_path)
with torch.no_grad():
    reloaded_output = reloaded_model(inputs['input_ids'], inputs['attention_mask'])

print(sim(reloaded_output[0].numpy(), logits.numpy()))
