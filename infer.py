import os
import time

import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import sounddevice as sd
from util import list_files

import numpy as np

print("Loading model...")
config = XttsConfig()

model_config_path = r'E:\Data\Models\XTTS-v2\config.json'
model_path = r'E:\Data\Models\XTTS-v2'
config.load_json(model_config_path)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=model_path, use_deepspeed=False)
model = model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
    audio_path=[r'gwy.wav',r'gwy2.wav'])

print("Inference...")
out = model.inference(
    """
    子贡问曰：“赐也何如？”子曰：“女器也。”曰：“何器也？”曰：“瑚琏也。”
    或曰：“雍也仁而不佞。”子曰：“焉用佞？御人以口给，屡憎于人。不知其仁，焉用佞？”
    子使漆雕开仕，对曰：“吾斯之未能信。”子说。
    子曰：“道不行，乘桴浮于海，从我者其由与？”子路闻之喜，子曰：“由也好勇过我，无所取材。”
    """,
    "zh-cn",
    gpt_cond_latent,
    speaker_embedding,
    temperature=0.7,  # Add custom parameters here
)

sample_rate = 24000
#
speech_data = np.array(out["wav"])
sd.play(speech_data, samplerate=sample_rate)
sd.wait()  # 等待播放结束

torchaudio.save("xtts.wav", torch.tensor(out["wav"]).unsqueeze(0), sample_rate, encoding='PCM_S')

print("Inference...")
t0 = time.time()
chunks = model.inference_stream(
    "你怎么这么自私。有没有见过黑社会啊。",
    "zh-cn",
    gpt_cond_latent,
    speaker_embedding
)

wav_chuncks = []
for i, chunk in enumerate(chunks):
    if i == 0:
        print(f"Time to first chunck: {time.time() - t0}")
    print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
    wav_chuncks.append(chunk)
wav = torch.cat(wav_chuncks, dim=0)
torchaudio.save("xtts_streaming.wav", wav.squeeze().unsqueeze(0).cpu(), 24000, format='wav', encoding='PCM_S')
