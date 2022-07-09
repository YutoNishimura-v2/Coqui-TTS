# 音声とそのテキストが入った json ファイルを用いて asr によるスコアリングを行う
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict
import logging
from logging import Logger

import Levenshtein
import speech_recognition as sr
from tqdm import tqdm

from TTS.tts.utils.text.cleaners import english_cleaners

###################################################
json_path = "out/text_data.json"
wav_base  = "out/"
output_path = "out/"
n_jobs = 20
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


def asr_and_scoring(path: Path, recognizer: sr.Recognizer, target_text: str, logger: Logger):
    try:
        predicted = speech2text(recognizer, str(path))
        score = calc_score(predicted, target_text)
    except KeyboardInterrupt:
        exit(0)
    except:
        logger.error("\nasr error", exc_info=True)
        logger.info(f"error path: {str(path)}")
        logger.info(f"error target_text: {target_text}")
        return None, None, None
    return path, predicted, score


# json 読み込み
with open(json_path, "r") as f:
    text_data = json.load(f)

# 音声パスリスト作成
wav_pathes = list(Path(wav_base).glob("**/*.wav"))

# logger の用意
logger = logging.getLogger("for error detection")
logger.setLevel(logging.DEBUG)
handler1 = logging.StreamHandler()
handler1.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s %(message)s"))
handler2 = logging.FileHandler(filename=Path(output_path) / "asr_erros.log")
handler2.setLevel(logging.WARN)
handler2.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s %(message)s"))
logger.addHandler(handler1)
logger.addHandler(handler2)

# 並列実行
print("process started")
result_data = {}
with ProcessPoolExecutor(n_jobs) as executor:
    futures = [
        executor.submit(
            asr_and_scoring,
            wav_path,
            recognizer,
            get_text_from_path(wav_path, text_data),
            logger,
        )
        for wav_path in wav_pathes
    ]
    for future in tqdm(futures):
        path, predicted, score = future.result()
        if predicted is None:
            continue
        target, spk, train_dev, _id = get_text_from_path(path, text_data, True)

        if spk not in result_data.keys():
            result_data[spk] = {}
        if train_dev not in result_data[spk].keys():
            result_data[spk][train_dev] = {}
        result_data[spk][train_dev][_id] = {}
        result_data[spk][train_dev][_id]["target"] = target
        result_data[spk][train_dev][_id]["predicted"] = predicted
        result_data[spk][train_dev][_id]["score"] = score

# for wav_path in tqdm(wav_pathes):
#     path, predicted, score = asr_and_scoring(
#         wav_path,
#         recognizer,
#         get_text_from_path(wav_path, text_data),
#         logger,
#     )
#     if predicted is None:
#         continue
#     target, spk, train_dev, _id = get_text_from_path(path, text_data, True)

#     if spk not in result_data.keys():
#         result_data[spk] = {}
#     if train_dev not in result_data[spk].keys():
#         result_data[spk][train_dev] = {}
#     result_data[spk][train_dev][_id] = {}
#     result_data[spk][train_dev][_id]["target"] = target
#     result_data[spk][train_dev][_id]["predicted"] = predicted
#     result_data[spk][train_dev][_id]["score"] = score


with open(f"{output_path}/results.json", 'w') as f:
    json.dump(result_data, f, indent=4)
