# 使い方
JSUTで書いていたけど，バックアップミスによって消し飛ばしてしまったので再度記載
やること順に記載していく

## resampleする
- 前処理として16000Hzにする必要あり
- downsampling.ipynb を動かすだけ

## configを途中まで書く
変更点は以下(issueで配布されているconfigからの変更点)
- use_language_weighted_sampler = false
    - 日本語しか使わないので一応falseに
- output_path: "checkpoints/japanese_multispeaker_exp_20220228/"
- use_phonemes: true
    - 日本語用
- phoneme_language: "ja-jp"
  - これと全く同一名でないと場合分けに引っかからないので注意
- phoneme_cache_path: "exps/japanese_multispeaker_exp_20220228/phoneme_cache/"
- min_seq_len: 1
    - 日本語は音素ではなくひらがなで数えるので．
- datasets: 適当．format書きながら確かめつつ
- test_sentences: 適当．出力したい文章
- use_speaker_embedding: true
- d_vector_file: 後でspeaker作るので，その保存予定のパスを書いておく
- num_languages: 1
- speaker_encoder_config_path: 自分で置いた場所

## formatを書く
TTS/tts/datasets/formatters.py
ここに，出力が以下の形式になるように関数を定義する
```python
[テキスト, wav_path, speaker_name]: こうするだけ．
例: (VCTK)
['Gerhard Schroeder was the victor.\n', 'datasets/VCTK-Corpus-removed-silence_16Khz/wav48/p241/p241_056.wav', 'VCTK_p241']
['They will do their own thing.\n', 'datasets/VCTK-Corpus-removed-silence_16Khz/wav48/p241/p241_166.wav', 'VCTK_p241']
['This is a war over our home.\n', 'datasets/VCTK-Corpus-removed-silence_16Khz/wav48/p241/p241_301.wav', 'VCTK_p241']
```
動かすためのコードはこんな感じ
- `export PYTHONPATH=/home/yuto_nishimura/workspace/python/yellston/TTS`
- `python3 TTS/bin/train_tts.py --config_path exps/japanese_multispeaker_exp_20220228/config.json`
これで動かしながらコードを書きましょう

## speaker embを作成する
- コードを回すだけ
- `python3 TTS/bin/compute_embeddings.py exps/japanese_multispeaker_exp_20220228/model_se.pth.tar exps/japanese_multispeaker_exp_20220228/config_se.json exps/japanese_multispeaker_exp_20220228/config.json exps/japanese_multispeaker_exp_20220228/d_vector_file.json`


## trainする
- コードは上と同じ
- min_seq_len = 11 にした．これはtrainコードでエラーが出ないギリギリ

- finetuningするので，
- `python3 TTS/bin/train_tts.py --config_path exps/japanese_multispeaker_exp_20220228/config.json --restore_path exps/tts_models--multilingual--multi-dataset--your_tts/model_file.pth.tar`