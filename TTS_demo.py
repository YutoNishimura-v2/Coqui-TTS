# アクセントに対応したdemoはないので新規作成
# 固定speaker id にも未対応だったので対応

import os
import string

import torch

from TTS.tts.utils.synthesis import synthesis
try:
  from TTS.utils.audio import AudioProcessor
except:
  from TTS.utils.audio import AudioProcessor


from TTS.tts.models import setup_model
from TTS.config import load_config
from TTS.tts.models.vits import *


###################################################
OUT_PATH = 'out/'
MODEL_PATH = 'checkpoints/20220429_pro_eng_3/20220429_pro_eng_3-May-08-2022_06+42PM-cf65193d/checkpoint_90000.pth.tar'
CONFIG_PATH = 'checkpoints/20220429_pro_eng_3/20220429_pro_eng_3-May-08-2022_06+42PM-cf65193d/config.json'
TTS_LANGUAGES = "checkpoints/20220429_pro_eng_3/20220429_pro_eng_3-May-08-2022_06+42PM-cf65193d/language_ids.json"
TTS_SPEAKERS = "checkpoints/20220429_pro_eng_3/20220429_pro_eng_3-May-08-2022_06+42PM-cf65193d/speaker_ids.json"

SPEAKER_ID = 248
LANGUAGE_ID = 0
LANGUAGE = "en"  # ["ja-jp", "zh-CN", "en"]
text = "All in the golden afternoon Full leisurely we glide; For both our oars, with little skill, By little arms are plied, While little hands make vain pretence Our wanderings to guide."
accent = None
###################################################
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

if SPEAKER_ID is None:
    print("select from below")
    print(model.speaker_manager.speaker_ids)
    SPEAKER_ID = int(input())

if LANGUAGE_ID is None:
    print("select from below")
    print(model.language_manager.language_id_mapping)
    LANGUAGE_ID = int(input())

print(" > text: {}".format(text))
wav, alignment, _, _, _ = synthesis(
                    model,
                    text,
                    accent,
                    LANGUAGE,
                    C,
                    "cuda" in str(next(model.parameters()).device),
                    ap,
                    speaker_id=SPEAKER_ID,
                    language_id=LANGUAGE_ID,
                    enable_eos_bos_chars=C.enable_eos_bos_chars,
                    use_griffin_lim=True,
                    do_trim_silence=False,
                ).values()
file_name = text.replace(" ", "_")
file_name = file_name.translate(str.maketrans('', '', string.punctuation.replace('_', ''))) + '.wav'
out_path = os.path.join(OUT_PATH, file_name)
print(" > Saving output to {}".format(out_path))
ap.save_wav(wav, out_path)
