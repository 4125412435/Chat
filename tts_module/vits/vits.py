# coding=utf-8
import json
import os.path

import numpy as np
from torch import no_grad, LongTensor

import tts_module.vits.commons as commons
import tts_module.vits.utils as utils
from tts_module.vits.models import SynthesizerTrn
from tts_module.vits.text import text_to_sequence, _clean_text


class ViTs:
    def __init__(self, config_path, models_info_path, models_path, device='cuda'):
        self.hps_ms = utils.get_hparams_from_file(config_path)
        self.models_info_path = models_info_path
        self.models_path = models_path
        with open(models_info_path, "r", encoding="utf-8") as f:
            self.models_info = json.load(f)
        self.model = None
        self.device = device

    def load_model(self, name):
        for i, info in self.models_info.items():
            sid = info['sid']
            name_en = info['name_en']
            name_zh = info['name_zh']
            if name_zh != name:
                continue
            title = info['title']
            cover = f"{self.models_path}/{i}/{info['cover']}"
            example = info['example']
            language = info['language']
            net_g_ms = SynthesizerTrn(
                len(self.hps_ms.symbols),
                self.hps_ms.data.filter_length // 2 + 1,
                self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
                n_speakers=self.hps_ms.data.n_speakers if info['type'] == "multi" else 0,
                **self.hps_ms.model)
            utils.load_checkpoint(f'{self.models_path}/{i}/{i}.pth', net_g_ms, None)
            net_g_ms = net_g_ms.eval().to(self.device)
            self.model = (
                sid, name_en, name_zh, title, cover, example, language, net_g_ms, self.create_tts_fn(net_g_ms, sid),
                self.create_to_symbol_fn(self.hps_ms))

    def get_text(self, text, hps, is_symbol):
        text_norm, clean_text = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = LongTensor(text_norm)
        return text_norm, clean_text

    def create_tts_fn(self, net_g_ms, speaker_id):
        def tts_fn(text, language, noise_scale, noise_scale_w, length_scale, is_symbol):
            text = text.replace('\n', ' ').replace('\r', '').replace(" ", "")
            if not is_symbol:
                if language == 0:
                    text = f"[ZH]{text}[ZH]"
                elif language == 1:
                    text = f"[JA]{text}[JA]"
                else:
                    text = f"{text}"
            stn_tst, clean_text = self.get_text(text, self.hps_ms, is_symbol)
            with no_grad():
                x_tst = stn_tst.unsqueeze(0).to(self.device)
                x_tst_lengths = LongTensor([stn_tst.size(0)]).to(self.device)
                sid = LongTensor([speaker_id]).to(self.device)
                audio = \
                    net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                                   length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()

            return "Success", (22050, audio)

        return tts_fn

    def create_to_symbol_fn(self, hps):
        def to_symbol_fn(is_symbol_input, input_text, temp_lang):
            if temp_lang == 0:
                clean_text = f'[ZH]{input_text}[ZH]'
            elif temp_lang == 1:
                clean_text = f'[JA]{input_text}[JA]'
            else:
                clean_text = input_text
            return _clean_text(clean_text, hps.data.text_cleaners) if is_symbol_input else ''

        return to_symbol_fn

    def change_lang(self, language):
        if language == 0:
            return 0.6, 0.668, 1.2
        elif language == 1:
            return 0.6, 0.668, 1
        else:
            return 0.6, 0.668, 1

    # sample_rate = 22050
    def generate_speech(self, text, ns=0.6, nsw=0.668, ls=0.95):
        lang = 2
        symbol_input = False
        sid, name_en, name_zh, title, cover, example, language, net_g_ms, tts_fn, to_symbol_fn = self.model
        o1, o2 = tts_fn(text, lang, ns, nsw, ls, symbol_input)
        wav = np.array(o2[1])
        return wav
