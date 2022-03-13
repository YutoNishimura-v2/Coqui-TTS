# train_memo
動かすためのコードはこんな感じ
- `export PYTHONPATH=/home/yuto_nishimura/workspace/python/yellston/TTS`
- `python3 TTS/bin/train_tts.py --config_path exps/20220309_japanese_multispeaker_wo_speakerencoder_exp/config.json --restore_path exps/tts_models--multilingual--multi-dataset--your_tts/model_file.pth.tar`
- `python3 TTS/bin/compute_embeddings.py exps/japanese_jsut_20220308/model_se.pth.tar exps/japanese_jsut_20220308/config_se.json exps/japanese_jsut_20220308/config.json exps/japanese_jsut_20220308/d_vector_file.json`

## exp
- 20220228_japanese_multispeaker_exp
  - とりあえず，JSUT+アリアルミリアルアベルーニの読み上げ+感情の合計16話者日本語のみのfinetuning
  - 配布されている重みからのスタート
  - 1000epoch
  - 結果:
    - 微妙
    - 発話が安定するまでも500epochくらいかかってた
    - 1000epochでも話者性は安定せず
- 20220308_japanese_jsut
  - JSUTのみで回してみる．
  - 配布されている重みからのスタート
  - 100epoch
  - 結果:
    - 普通に良い感じ
    - さすがに単一話者は余裕
- 20220309_japanese_multispeaker_wo_speakerencoder_exp
  - 20220228_japanese_multispeaker_exp と同じデータだけど，Speaker encoderを利用せずに実行してみる
    - 具体的には，use_d_vector系をnullとかにするだけで勝手にspeaker ID作ってくれる
  - 配布重みからスタート
  - 1000epoch
  - 結果:
