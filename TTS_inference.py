# formatter を用いて，dataset のデータを推論して出力する．
import json
import logging
import os
import string
from pathlib import Path
from typing import Dict, List, Optional

import torch

from TTS.tts.utils.synthesis import synthesis

try:
    from TTS.utils.audio import AudioProcessor
except:
    from TTS.utils.audio import AudioProcessor

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
        "path": "/mnt/nas/disk1_4tb/dataset/LibriTTS-16khz/",
        "meta_file_train": None,
        "ununsed_speakers": None,
        "language": "en",
        "meta_file_val": None,
        "meta_file_attn_mask": None,
    },
    {
        "name": "ljspeech",
        "path": "/mnt/nas/disk1_4tb/dataset/LJSpeech-1.1/",
        "meta_file_train": "metadata.csv",
        "ununsed_speakers": None,
        "language": "en",
        "meta_file_val": None,
        "meta_file_attn_mask": None,
    },
    {
        "name": "vctk",
        "path": "/mnt/nas/disk1_4tb/dataset/VCTK-Corpus/",
        "meta_file_train": None,
        "ununsed_speakers": None,
        "language": "en",
        "meta_file_val": None,
        "meta_file_attn_mask": None,
    },
    {
        "name": "common_voice",
        "path": "/mnt/nas/disk1_4tb/dataset/cv-corpus-9.0-2022-04-27/en/",
        "meta_file_train": "train.tsv",
        "ununsed_speakers": None,
        "language": "en",
        "meta_file_val": "validated.tsv",
        "meta_file_attn_mask": None,
    },
    {
        "name": "coefont_english_7000",
        "path": "/mnt/nas/disk1_4tb/dataset/transcripts/",
        "meta_file_train": None,
        "ununsed_speakers": None,
        "language": "en",
        "meta_file_val": None,
        "meta_file_attn_mask": None,
    }
]
# use_speakers: Optional[List[str]] = [
#     '0755a9ad-d7dc-4dd2-99d0-0590b90f2c60', '0770613f-0a0c-49c1-94b8-80823793675a', '0e28fe50-fe7f-4bc6-a043-8a49f9d0a583', '2726f8ee-141c-4df0-8867-73e386359aee', '272af3ed-71c9-4e83-b66f-5ff22c08fae6', '28754739-f751-4a73-9d69-4066db35ee68', '2b802cd1-868f-4391-9129-96335d719764', '2f26f1b2-4758-4ead-9c7b-42a438238ef7', '381d2d80-ddfe-4a44-8572-e94870dfc329', '3d86ade1-c095-4d53-ab3c-d4264b020e06', '44d9280e-0d0e-4fcf-813c-b1851cf17cef', '45321ffe-7838-4d71-bdf9-153310bc959c', '4dc53189-840f-42f7-8827-9898ce468642', '4ed445d5-f04c-4f5b-86f4-15b230032db4', '5bf09952-6b1c-454d-bbdc-3bf2c59698fb', '66540e9e-197b-4b20-9f11-249feb2cfe04', '684e3b78-fd6b-4fd1-b943-e82c4068c47a', '6c48173c-cdc7-4e13-a52d-357425666a05', '718039cb-26b9-4686-8b88-e09f7ae4a7e8', '7450f1f6-8043-4b24-b52c-cce389662d55', '7519b285-eda9-4555-bc7c-09a6983d830f', '806b2526-2ab1-479b-9e6e-36202198752a', '858907a7-8f02-427c-bd6a-b7d4650abd24', '8f034d8c-954e-4cdb-a219-e9b4367048e1', '958cebba-b772-4e31-b697-218d792cacb6', '96ba9521-7657-413b-8786-eb1205775a94', '9a7ffd0f-5968-456c-a5be-3efb1600e4db', '9eaa0a8f-16ca-4fa7-838f-7cfcd6804543',
#     'a260794a-6653-47b1-b780-60453a2b8762', 'a4dcb6a7-0704-40ac-a18d-4366e9c9c230', 'a6c02557-80f8-4b47-aa8e-ffe13b5e16e8',
#     'allial_ikari', 'allial_kanashimi', 'allial_normal', 'allial_tanoshimi', 'allial_yorokobi', 'averuni_ikari', 'averuni_kanashimi', 'averuni_normal', 'averuni_tanoshimi', 'averuni_yorokobi',
#     'b0d29da7-412a-4fa4-ad2c-1a49ec70fb50', 'b1aee780-2e71-4af7-9660-ac3e4af6df48', 'b1cd60fa-e013-410a-8fe7-1c89773c2f88', 'b99f765c-4e63-4fb7-b53f-1deb5447555e', 'bcc347fa-abff-461f-abab-b69d6140903a', 'c0b5de35-7d82-4045-93e6-f7e215e309b4', 'c1248860-738f-4c37-bb2e-6daa1792997a', 'c1e18405-2e02-412c-b3f4-86e4b27b9329', 'c2b47a6d-f636-47e9-9184-5fa59754dd75', 'c3c5626c-8714-4615-a5bc-8f4ab5ef4ebe', 'c5375a07-e6b9-437b-a4ee-cb6439a0d95c', 'c589499f-d5aa-4ca4-8c4a-1ac315554ae9', 'cad9f33f-7d70-425f-bf62-5d220d94c970', 'd315c76f-1028-49e6-ba3a-daa2927912f7', 'd6740fe4-cf25-4fa4-8b06-2735edc86e97', 'e0aefd54-8e45-473d-b88b-96ec64b99c71', 'e0ec927a-5ce9-4a70-94cd-e64226bc94a0', 'e80cc248-bd23-4ee3-af62-8740342262ec', 'ec5a075c-3f7f-4992-b3b0-83854486a157', 'edd6cc3e-c2b1-451e-bea0-ac6954e99488', 'f1ab8776-ea64-45cd-b815-2d5e2b91de86', 'f38d122c-6bec-4eb5-bde1-303c2bb627ef', 'f4acd312-41cb-4ea0-8999-818315c4e48b', 'f5cc38db-6096-438a-8877-c7651a1894ec', 'f7e6457e-bffe-4deb-ab94-f439bf2c6737', 'ff255f62-fded-4879-8808-6bde5260b48f',
#     'fujisaki', 'millial_ikari', 'millial_kanashimi', 'millial_normal', 'millial_tanoshimi', 'millial_yorokobi', 'morikawa']
use_speakers: Optional[List[str]] = ['averuni_normal', 'fujisaki', 'morikawa']
use_languages: Optional[List[str]] = ["en"]  # None で全ての言語を出力する
n_jobs = 10
overwrite = True
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
# logger の設定
logger = logging.getLogger("for error detection")
logger.setLevel(logging.DEBUG)
handler1 = logging.StreamHandler()
handler1.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s %(message)s"))
handler2 = logging.FileHandler(filename=Path(OUT_PATH) / "synthesis_errors.log")
handler2.setLevel(logging.WARN)
handler2.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s %(message)s"))
logger.addHandler(handler1)
logger.addHandler(handler2)


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
        print("total item num: ", len(items))
        _id = 0
        for text, accent, _, _, lang in tqdm(items, leave=False):
            file_name = text.replace(" ", "_")[:50]
            file_name = file_name.translate(str.maketrans('', '', string.punctuation.replace('_', ''))) + '.wav'
            out_path = output_base / (str(_id)+"_"+file_name)
            _id += 1
            if overwrite is False and out_path.exists() is True:
                continue
            try:
                text, wav, _id = _synthesis(
                    model, text, accent, speaker, lang, C, ap,
                    model.speaker_manager.speaker_ids,
                    model.language_manager.language_id_mapping,
                    use_languages, _id
                )
                ap.save_wav(wav, out_path)
                save_text_data[speaker][train_eval][_id] = text

            except KeyboardInterrupt:
                exit(0)
            except:
                logger.error("\nsynthesis error", exc_info=True)
                logger.info(f"error text: {text}")
                logger.info(f"error accent: {accent}")
                logger.info(f"error speaker: {speaker}")
                logger.info(f"error lang: {lang}")
                logger.info(f"error id: {_id}")

        with open(f"{OUT_PATH}/text_data.json", 'w') as f:
            json.dump(save_text_data, f, indent=4)
