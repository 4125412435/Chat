import os.path

import langid

import pandas as pd
from pathlib import Path

from whisper import Whisper
from util import list_files


def gen_dataset(stt_model, audio_paths, out_path):
    text_output = stt_model.translate(audio_paths, language='chinese')
    langs = [langid.classify(text) for text in text_output]
    wav_names = [Path(audio).stem for audio in audio_paths]
    result = []

    for text, lang, name in zip(text_output, langs, wav_names):
        result.append({
            'filename': name,
            'text': text,
            'normalized_text': text,
        })
    df = pd.DataFrame(result)
    df.to_csv(out_path, sep='|', header=False, index=False)


whisper_model = Whisper('E:/Data/Models/whisper-large-v3-turbo', 'cuda')

audio_dir = '../resources/audio/alice_zh'
metadata_path = 'metadata.csv'
gen_dataset(whisper_model, list_files(audio_dir, 'wav'), out_path=os.path.join(audio_dir,metadata_path))
