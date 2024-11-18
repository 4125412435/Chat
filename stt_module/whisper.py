from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa


# Function to load and preprocess the audio file
def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=16000)  # Resample to 16000 Hz
    return audio


def resample_audio(audio_frames, origin_rate):
    return librosa.resample(audio_frames, orig_sr=origin_rate, target_sr=16000)


class Whisper:
    def __init__(self, model_path, device='cuda'):
        # Path to your model folder

        # Load the Whisper model directly from the local folder
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
        self.device = torch.device(device)
        # Load Whisper processor for feature extraction and tokenization
        self.processor = WhisperProcessor.from_pretrained(model_path)

    def translate_audio_file(self, audio_paths, max_batch=10, language=None):
        # Path to your .wav file
        if not isinstance(audio_paths, list):
            audio_paths = [audio_paths]

        audio_input = [load_audio(wav_file_path) for wav_file_path in audio_paths]

        return self.translate(audio_input, max_batch, language)

    def translate(self, audio_input, origin_rate=44100, max_batch=10, language=None):
        if not isinstance(audio_input, list):
            audio_input = [audio_input]

        audio_input = [resample_audio(audio, origin_rate) for audio in audio_input]
        result = []

        for i in tqdm(range(0, len(audio_input), max_batch)):
            # Process the audio
            input_features = self.processor(audio_input[i:min(i + max_batch, len(audio_input))], sampling_rate=16000,
                                            return_tensors="pt").input_features.to(
                self.device)

            # Perform translation
            with torch.no_grad():
                generated_ids = self.model.generate(input_features, language=language)
                translated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                for text in translated_text:
                    result.append(text)
        return result
