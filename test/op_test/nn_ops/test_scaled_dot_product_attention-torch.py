import torch
import torch.nn as nn
import numpy as np
from utils.run import run_parser


class scaled_dot_product_attention_model(nn.Module):
    def __init__(self, dropout_p, scale):
        super(scaled_dot_product_attention_model, self).__init__()
        self.dropout_p = dropout_p
        self.scale = scale

    def forward(self, q, k, v, attn_mask):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask,
                                                                dropout_p=self.dropout_p,
                                                                scale=self.scale)


class scaled_dot_product_attention_with_causal_mask_model(nn.Module):
    def __init__(self, dropout_p, scale):
        super(scaled_dot_product_attention_with_causal_mask_model, self).__init__()
        self.dropout_p = dropout_p
        self.scale = scale

    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v,
                                                                dropout_p=self.dropout_p,
                                                                is_causal=True,
                                                                scale=self.scale)


def create_scaled_dot_product_attention_model(model_path, dropout_p, is_causal, scale):
    try:
        if is_causal:
            model = scaled_dot_product_attention_with_causal_mask_model(dropout_p, scale)
        else:
            model = scaled_dot_product_attention_model(dropout_p, scale)
        model_scripted = torch.jit.script(model)
        model_scripted.save(model_path)
    except Exception as e:
        print('Fail to create torch model because %s' % str(e))


TEST_NAME = 'scaled_dot_product_attention'

# prepare model and input datas
query = np.random.ranf([32, 8, 128, 64]).astype(np.float32)
key = np.random.ranf([32, 8, 128, 64]).astype(np.float32)
value = np.random.ranf([32, 8, 128, 64]).astype(np.float32)
attn_mask = np.random.ranf([32, 8, 128, 128]).astype(np.float32)
feed_dict = {'query': query, 'key': key, 'value': value}
# The output of the torch model is not determined when dropout_p is not 0
for dropout_p in (0.0, ):
    for is_causal in (True, False, ):
        if not is_causal:
            feed_dict.update({'attn_mask': attn_mask})
        elif 'attn_mask' in feed_dict:
            feed_dict.pop('attn_mask')
        for scale in (None, 1.2):
            model_path = '-'.join([TEST_NAME, str(dropout_p), str(is_causal), str(scale)]) + '.pt'
            create_scaled_dot_product_attention_model(model_path, dropout_p, is_causal, scale)
            exit_status = run_parser(model_path, feed_dict)
            assert exit_status
