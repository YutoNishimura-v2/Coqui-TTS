# 音声とそのテキストが入った json ファイルを用いて asr によるスコアリングを行う
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict

import Levenshtein
import speech_recognition as sr
from tqdm import tqdm

from TTS.tts.utils.text.cleaners import english_cleaners

###################################################
json_path = ""
wav_base  = ""
output_json_path = ""
n_jobs = 5
###################################################

recognizer = sr.Recognizer()
def speech2text(recognizer: sr.Recognizer, path: str) -> str:
    with sr.AudioFile(path) as source:
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio, language="en-US")


def calc_score(predicted: str, target: str):
    predicted = english_cleaners(predicted).replace(".", "").replace(",", "").replace("?", "").replace("!", "")
    target = english_cleaners(target).replace(".", "").replace(",", "").replace("?", "").replace("!", "")
    return Levenshtein.ratio(predicted, target)


def get_text_from_path(path: Path, text_data: Dict, need_detail: bool = False):
    _id = path.stem.split("_")[0]
    spk = path.parent.stem
    train_dev = path.parent.parent.stem
    if need_detail is True:
        return text_data[spk][train_dev][_id], spk, train_dev, _id
    return text_data[spk][train_dev][_id]


def asr_and_scoring(path: Path, recognizer: sr.Recognizer, target_text: str):
    predicted = speech2text(recognizer, str(path))
    score = calc_score(predicted, target_text)
    return path, predicted, score


# json 読み込み
with open(json_path, "r") as f:
    text_data = json.load(f)

# 音声パスリスト作成
wav_pathes = Path(wav_base).glob("**/*.wav")

# 並列実行
result_data = {}
with ProcessPoolExecutor(n_jobs) as executor:
    futures = [
        executor.submit(
            asr_and_scoring,
            wav_path,
            recognizer,
            get_text_from_path(wav_path, text_data)
        )
        for wav_path in wav_pathes
    ]
    for future in tqdm(futures):
        path, predicted, score = future.result()
        target, spk, train_dev, _id = get_text_from_path(path, text_data, True)

        if spk not in result_data.keys():
            result_data[spk] = {}
        if train_dev not in result_data[spk].keys():
            result_data[spk][train_dev] = {}
        result_data[spk][train_dev][_id] = {}
        result_data[spk][train_dev][_id]["target"] = target
        result_data[spk][train_dev][_id]["predicted"] = predicted
        result_data[spk][train_dev][_id]["score"] = score

with open(output_json_path, 'w') as f:
    json.dump(result_data, f, indent=4)
