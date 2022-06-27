# formatter を用いて，dataset のデータを推論して出力する．
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from this import d

from TTS.tts.utils.synthesis import synthesis

try:
    from TTS.utils.audio import AudioProcessor
except:
    from TTS.utils.audio import AudioProcessor

from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from TTS.config import load_config, register_config
from TTS.tts.datasets import load_meta_data
from TTS.tts.models import setup_model
from TTS.tts.models.vits import *

###################################################
OUT_PATH: str = 'out/'
MODEL_PATH: str = 'checkpoints/20220609_pro_studio_light/20220609_pro_studio_light-June-22-2022_05+28PM-ef04a15e/checkpoint_265000.pth.tar'
CONFIG_PATH: str = 'checkpoints/20220609_pro_studio_light/20220609_pro_studio_light-June-22-2022_05+28PM-ef04a15e/config.json'
TTS_LANGUAGES: str = "checkpoints/20220609_pro_studio_light/20220609_pro_studio_light-June-22-2022_05+28PM-ef04a15e/language_ids.json"
TTS_SPEAKERS: str = "checkpoints/20220609_pro_studio_light/20220609_pro_studio_light-June-22-2022_05+28PM-ef04a15e/speaker_ids.json"

model_name: str = "vits"
# config と同じ定義の仕方をする．但し，null ではなく None を使う．
datasets: List[Dict[str, str]] = [
    {
        "name": "libri_tts",
        "path": "../dataset/LibriTTS-16khz/",
        "meta_file_train": None,
        "ununsed_speakers": None,
        "language": "en",
        "meta_file_val": None,
        "meta_file_attn_mask": None,
    }
]
use_speakers: Optional[List[str]] = None  # None で全員で出力する
use_languages: Optional[List[str]] = ["en"]  # None で全ての言語を出力する
n_jobs = 10
###################################################
"""dataset setup"""
print("load dataset metadata...")
# class を合わせるため，空の config を読み込む.
config_class = register_config(model_name.lower())
config = config_class()
config.from_dict({"datasets": datasets})
data_train, data_eval = load_meta_data(config.datasets)

"""model setup"""
USE_CUDA = torch.cuda.is_available()
os.makedirs(OUT_PATH, exist_ok=True)

# load the config
C = load_config(CONFIG_PATH)

# load the audio processor
ap = AudioProcessor(**C.audio)

model = setup_model(C)
model.language_manager.set_language_ids_from_file(TTS_LANGUAGES)
model.speaker_manager.set_speaker_ids_from_file(TTS_SPEAKERS)

cp = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
# remove speaker encoder
model_weights = cp['model'].copy()
for key in list(model_weights.keys()):
    if "speaker_encoder" in key:
        del model_weights[key]

model.load_state_dict(model_weights)
model.eval()

if USE_CUDA:
    model = model.cuda()

"""inference"""
def _synthesis(model, text, accent, speaker, lang, config, audio_processor, spk2id, lang2id, use_langages, _id):
    if use_langages is not None:
        if lang not in use_langages:
            return None, None, None
    wav, _, _, _, _ = synthesis(
        model,
        text,
        accent,
        lang,
        config,
        "cuda" in str(next(model.parameters()).device),
        audio_processor,
        speaker_id=int(spk2id[speaker]),
        language_id=int(lang2id[lang]),
        enable_eos_bos_chars=C.enable_eos_bos_chars,
        use_griffin_lim=True,
        do_trim_silence=False,
    ).values()
    return text, wav, _id


speakers = use_speakers
if use_speakers is None:
    speakers = model.speaker_manager.speaker_ids.keys()

save_text_data = {}
for speaker in tqdm(speakers):
    save_text_data[speaker] = {}
    for i, items in enumerate([data_train, data_eval]):
        if i == 0:
            train_eval = "train"
        else:
            train_eval = "eval"
        output_base = Path(OUT_PATH) / train_eval / speaker
        save_text_data[speaker][train_eval] = {}

        output_base.mkdir(parents=True, exist_ok=True)

        with ProcessPoolExecutor(n_jobs) as executor:
            futures = [
                executor.submit(
                    _synthesis,
                    model, text, accent, speaker, lang, C, ap,
                    model.speaker_manager.speaker_ids, 
                    model.language_manager.language_id_mapping,
                    use_languages, _id
                )
                for _id, (text, accent, _, _, lang) in enumerate(items)
            ]
            for future in tqdm(futures, leave=False):
                text, wav, _id = future.result()
                out_path = os.path.join(output_base, str(_id)+".wav")
                ap.save_wav(wav, out_path)
                save_text_data[speaker][train_eval][_id] = text


with open(f"{OUT_PATH}/text_data.json", 'w') as f:
    json.dump(save_text_data, f, indent=4)
