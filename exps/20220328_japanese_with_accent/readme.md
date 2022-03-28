export PYTHONPATH="/home/yuto_nishimura/workspace/python/yellston/TTS:$PYTHONPATH"
python3 TTS/bin/train_tts.py \
    --config_path exps/20220328_japanese_with_accent/config.json \
    --restore_path exps/tts_models--multilingual--multi-dataset--your_tts/model_file.pth.tar

# アクセント追加するために変更した部分を記録しておく
- config
  - BaseTTSConfig:
    - 「"use_accent_info": true」
  - CharactersConfig:
    - 「"accents": "012"」
  - VitsArgs:
    - 「"use_accent_embedding": true」
    - 「"embedded_accent_dim": 256」

## memo
- config
  - 「distributed_url」まで:「BaseTrainingConfig」で管理
  - 「d_vector_dim」まで: 「BaseTTSConfig」で管理 (上のクラスを継承)
  - 残り物: 「vits_config.py」で管理していることに注意
    - 特に「model_args」は 「VitsArgs」で管理
  
  - 追加は上の構成に従ってやってみる．
