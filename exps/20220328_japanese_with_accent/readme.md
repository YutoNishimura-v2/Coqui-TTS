export PYTHONPATH="/home/yuto_nishimura/workspace/python/yellston/TTS:$PYTHONPATH"
python3 TTS/bin/train_tts.py \
    --config_path exps/20220328_japanese_with_accent/config.json \
    --restore_path exps/tts_models--multilingual--multi-dataset--your_tts/model_file.pth.tar

# アクセント追加するために変更した部分を記録しておく
- config
  - BaseTTSConfig:
    - 「"use_IPAg2p_phonemes": true」
    - 「"use_accent_info": true」
  - CharactersConfig:
    - 「"accents": "012"」
  - VitsArgs:
    - 「"use_accent_embedding": true」
    - 「"embedded_accent_dim": 256」

- formatters.py
  - coefont_studio
    - 「raw_text = raw_text + "_<accent>_" + accent_info」
    - ↑この形でアクセント情報を追加

## memo
- config
  - 「distributed_url」まで:「BaseTrainingConfig」で管理
  - 「d_vector_dim」まで: 「BaseTTSConfig」で管理 (上のクラスを継承)
  - 残り物: 「vits_config.py」で管理していることに注意
    - 特に「model_args」は 「VitsArgs」で管理
  
  - 追加は上の構成に従ってやってみる．

- accent
  - IPAg2pを利用する
  - 日本語のみアクセント情報が追加で必要．
  - それ以外の言語は不要
  - テキストの後半にアクセント情報も付記する形で保持する
    - at 「formatters.py」
    - そうすることで今までの形式は大きく変える必要をなくす
