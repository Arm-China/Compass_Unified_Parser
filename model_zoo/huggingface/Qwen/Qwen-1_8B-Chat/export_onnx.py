import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

remote_dir = '/project/ai/zhouyi_compass/Model_Resource/huggingface/Qwen/Qwen-1_8B-Chat'  # nopep8
sys.path.append(remote_dir)  # nopep8

ONNX_DIR = os.path.join(remote_dir, 'OUTPUTS/ONNX_WITH_ATTENTION_MASK')
os.makedirs(ONNX_DIR, mode=0o775, exist_ok=True)

device = 'cuda'
tokenizer = AutoTokenizer.from_pretrained(remote_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    remote_dir, device_map=device, trust_remote_code=True, fp32=True, **{'return_dict': False})
model = model.eval()

question = 'The Great Gatsby is a 1925 novel by American writer F. Scott Fitzgerald. Set in the Jazz Age on Long Island, near New York City, the novel depicts first-person narrator Nick Carraway\'s interactions with mysterious millionaire Jay Gatsby and Gatsby\'s obsession to reunite with his former lover, Daisy Buchanan. The novel was inspired by a youthful romance Fitzgerald had with socialite Ginevra King, and the riotous parties he attended on Long Island\'s North Shore in 1922. Following a move to the French Riviera, Fitzgerald completed a rough draft of the novel in 1924. He submitted it to editor Maxwell Perkins, who persuaded Fitzgerald to revise the work over the following winter. After making revisions, Fitzgerald was satisfied with the text, but remained ambivalent about the book\'s title and considered several alternatives. Painter Francis Cugat\'s dust jacket art, named Celestial Eyes, greatly impressed Fitzgerald, and he incorporated its imagery into the novel. After its publication by Scribner\'s in April 1925, The Great Gatsby received generally favorable reviews, though some literary critics believed it did not'   # nopep8

tokens_dict = tokenizer(question, return_tensor='pt')
response, history = model.chat(tokenizer, question, history=None)
print(response)

torch_tokens = torch.tensor([tokens_dict['input_ids']], device=device)
torch_atten_mask = torch.tensor([tokens_dict['attention_mask']], device=device)

traced_model = torch.jit.trace(model, example_kwarg_inputs={
                               'input_ids': torch_tokens, 'attention_mask': torch_atten_mask}, strict=True)


full_input_ids = [[151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 785, 8513, 479, 35514, 374, 264, 220, 16, 24, 17, 20, 11514, 553, 3693, 6916, 434, 13, 9815, 61214, 13, 2573, 304, 279, 35867, 13081, 389, 5724, 10720, 11, 3143, 1532, 4261, 4311, 11, 279, 11514, 61891, 1156, 28045, 64171, 14991, 3261, 1041, 352, 594, 21880, 448, 25382, 88944, 18919, 479, 35514, 323, 479, 35514, 594, 48535, 311, 34640, 632, 448, 806, 4741, 30557, 11, 70164, 84190, 13, 576, 11514, 572, 14606, 553, 264, 64555, 29263, 61214, 1030, 448, 3590, 632, 479, 482, 85, 956, 6210, 11, 323, 279, 41497, 782, 9677, 566, 18178, 389, 5724, 10720, 594, 4787, 44719, 304, 220, 16, 24, 17, 17, 13, 22713, 264, 3271, 311, 279, 8585,
                   50668, 25834, 11, 61214, 8145, 264, 11165, 9960, 315, 279, 11514, 304, 220, 16, 24, 17, 19, 13, 1260, 14634, 432, 311, 6440, 58397, 64911, 11, 879, 64001, 61214, 311, 64736, 279, 975, 916, 279, 2701, 12406, 13, 4636, 3259, 53762, 11, 61214, 572, 19527, 448, 279, 1467, 11, 714, 14616, 8873, 11769, 911, 279, 2311, 594, 2265, 323, 6509, 3807, 26450, 13, 96764, 25127, 356, 768, 266, 594, 15797, 26208, 1947, 11, 6941, 22687, 50606, 41996, 11, 18875, 24404, 61214, 11, 323, 566, 31662, 1181, 40445, 1119, 279, 11514, 13, 4636, 1181, 16599, 553, 75680, 65, 1194, 594, 304, 5813, 220, 16, 24, 17, 20, 11, 576, 8513, 479, 35514, 3949, 8789, 36749, 8379, 11, 3498, 1045, 31365, 22698, 11585, 432, 1521, 537, 151645, 198, 151644, 77091, 198]]
full_tokens = torch.tensor(full_input_ids, device=device)
full_mask = torch.ones_like(full_tokens)


do_constant_folding = False if device == 'cuda' else True
ONNX_PATH = os.path.join(ONNX_DIR, 'qwen_1_8b.onnx')
torch.onnx.export(traced_model, (full_tokens, full_mask), ONNX_PATH,
                  verbose=True, do_constant_folding=do_constant_folding)
