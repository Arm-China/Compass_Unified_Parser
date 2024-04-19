import sys                                                                                                       # nopep8
model_dir = '/project/ai/scratch01/guaxu01/eac_models/IR/code/model_zoo/Model_Resource/github/openai/whisper/'   # nopep8
sys.path.append(model_dir)                                                                                       # nopep8
import os                                                                                                        # nopep8
import torch                                                                                                     # nopep8
import whisper                                                                                                   # nopep8
from whisper.audio import SAMPLE_RATE, load_audio, log_mel_spectrogram                                           # nopep8
from whisper.decoding import DecodingOptions                                                                     # nopep8


torch.set_default_dtype(torch.float32)

model_name = 'small'
model_path = os.path.join(model_dir, 'SOURCES', model_name + '.pt')
audio_path = os.path.join(model_dir, 'SOURCES', 'neverendingstory-german.mp3')
TORCH_DIR = os.path.join(model_dir, 'OUTPUTS/TORCH')
os.makedirs(TORCH_DIR, mode=0o775, exist_ok=True)

model = whisper.load_model(model_path).to('cpu')
audio = load_audio(audio_path)

n_audio, n_mels, n_frames = 1, 80, 3000

mel_from_audio = log_mel_spectrogram(audio, n_mels=n_mels, padding=480000, device=torch.device('cpu'))
mel_from_audio_sliced = mel_from_audio[:, :3000]
mel_from_audio_reshaped = torch.reshape(mel_from_audio_sliced, (n_audio, n_mels, n_frames))
np_mel_from_audio_reshaped = mel_from_audio_reshaped.numpy()
options = DecodingOptions(language=None)
decode_result = model.decode(mel_from_audio_reshaped, options)

print(decode_result[0].language, 'decode text:',  decode_result[0].text)
print(len(decode_result[0].tokens), 'decode tokens:', decode_result[0].tokens)

traced_encoder = torch.jit.trace(model.encoder, torch.randn(n_audio, n_mels, n_frames))
encoder_path = os.path.join(TORCH_DIR, 'whisper_%s_encoder.pt' % model_name)
traced_encoder.save(encoder_path)

n_ctx, n_state = model.dims.n_audio_ctx, model.dims.n_audio_state
torch_tokens = torch.tensor([decode_result[0].tokens], dtype=torch.int32)
torch_audio_features = torch.randn(n_audio, n_ctx, n_state)
traced_decoder = torch.jit.trace(model.decoder, [torch_tokens, torch_audio_features])
decoder_path = os.path.join(TORCH_DIR, 'whisper_%s_decoder.pt' % model_name)
traced_decoder.save(decoder_path)
