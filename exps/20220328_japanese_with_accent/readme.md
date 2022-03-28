export PYTHONPATH="/home/yuto_nishimura/workspace/python/yellston/TTS:$PYTHONPATH"
python3 TTS/bin/train_tts.py \
    --config_path exps/20220328_japanese_with_accent/config.json \
    --restore_path exps/tts_models--multilingual--multi-dataset--your_tts/model_file.pth.tar

# アクセント追加するために変更した部分を記録しておく
- config
  - 「"use_accent_info": true」
  - 「"accents": "012"」
  - 「"use_accent_embedding": true」
  - 「"embedded_accent_dim": 256」

## memo
- config
  - ~「distributed_url」までが，「BaseTrainingConfig」で管理されている ~

