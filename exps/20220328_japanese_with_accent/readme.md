export PYTHONPATH="/home/yuto_nishimura/workspace/python/yellston/TTS:$PYTHONPATH"
python3 TTS/bin/train_tts.py \
    --config_path exps/20220328_japanese_with_accent/config.json \
    --restore_path exps/tts_models--multilingual--multi-dataset--your_tts/model_file.pth.tar

## 現状
テストセットを作るときにエラーが出てる
IPAにしてないぽい?

# アクセント追加するために変更した部分を記録しておく
- config
  - BaseTTSConfig:
    - 「"use_IPAg2p_phonemes": true」
    - 「"use_accent_info": true」
  - CharactersConfig:
    - 「"accents": "012"」
  - VitsArgs:
    - 「"use_accent_embedding": true」
    - 「"num_accents": 3」
    - 「"embedded_accent_dim": 256」

- formatters.py
  - coefont_studio
    - spkのあとにアクセント情報を追加
    - 最初は楽をするために「_<accent>_」なる文字を挟んでその後に入れようと思っていたけれど、汚いし、text cleanerによって除去されてしまう
      - そこで処理から外せば良いといえばそうだけど、ちょっと汚い。それなら改造点は多くなってもいいからなんとかちゃんともたせる。

## memo
- config
  - 「distributed_url」まで:「BaseTrainingConfig」で管理
  - 「d_vector_dim」まで: 「BaseTTSConfig」で管理 (上のクラスを継承)
  - 残り物: 「vits_config.py」で管理していることに注意
    - 特に「model_args」は 「VitsArgs」で管理
  
  - 追加は上の構成に従ってやってみる．

- accent
  - IPAg2pを利用する
    - python3.9だと、g2p-enとかを入れるときにエラーになる
      - pip install https://github.com/lancopku/pkuseg-python/archive/master.zip
      - これをすれば解決
      - これだけだとだめで、以下の措置をする
        - https://github.com/lancopku/pkuseg-python/releases/download/v0.0.16/mixed.zip
        - これをダウンロードして、以下のフォルダを作成し、解凍した中身をそこにおく
          - `/home/yuto_nishimura/miniconda3/envs/yourTTS/lib/python3.9/site-packages/pkuseg/models/default`
  - 日本語のみアクセント情報が追加で必要．
  - それ以外の言語は不要
  - テキストの直後に追加する
    - こうしないと、別の場所だといろいろ齟齬が起きるので。
  - configにおいて、phonemes にファイルパスを与えることでそこから音素リストを読み込めるようにした
    - 何をIDとして用いるか、はもしtts model側で「make_symbol」という関数を定義したらそれが最優先で利用される
      - なので、今回はそこでurlか否かを判断すると良さそう

- 処理されていることを順番に見ていく。
  - TTS/bin/train_tts.py
    - Trainer()
      - 「諸々の初期化を行っている」
      - 「modelとかoptimizer用意したり」
      - load_meta_data
        - 「ここで、langはitemの末尾へという要請」
      - Trainer.model.init_multispeaker
        - Vits.init_multispeaker
          - get_speaker_manager
            - SpeakerManager.set_speaker_ids_from_data
            - SpeakerManager.parse_speakers_from_data
              - 「ここで、itemの[2]がspkという仮定が入っている」
              - 「lenで場合分けしよう」
              - 「他にも、「item[」「self.items」で検索して変更の必要がある部分を変えていく」
      - Trainer.model.init_multispeaker
        - Vits.init_multilingual
          - get_language_manager
            - LanguageManager.set_language_ids_from_data
            - LanguageManager.parse_languages_from_data
              - 「ここもspeaker同様の調整」
    - Trainer.fit
    - Trainer._fit
    - Trainer.train_epoch
      - Trainer._get_loader
        - TTS/tts/models/base_tts.py
          - BaseTTS.get_data_loader
            - 「datasetのインスタンス化はここ」
            - 「ちゃんと追加した引数をここで反映すること」
            - 「compute_input_seq_cache = false にすることで、前もって音素にしてちゃんとseq lenをphoneme levelで数えてくれる」
              - 「今まではここをfalseにしていたせいで日本語は日本語で音素数を数えていた」
            - TTS/tts/datasets/TTSDataset.py
              - TTSDataset.compute_input_seq
                - 「ここで一旦phoneme onならphonemeの計算が入る」
                - TTSDataset._phoneme_worker
                  - 「ここで `text, wav_file, *_ = item` をやるので、itemの長さで場合分け。」
                  - TTSDataset._load_or_generate_phoneme_sequence
                    - TTSDataset._generate_and_cache_phoneme_sequence
                      - TTS/tts/utils/text/__init__.py
                        - phoneme_to_sequence
                          - 「まさにIPAに改造するべき部分。詳しく見ていく」
                          - _clean_text
                            - TTS/tts/utils/text/cleaners.py
                              - multilingual_cleaners
                                - 「テキストクリーナー」
                                - 「小文字にする→;とか&とかを別の文字に置き換え→カッコとかの除去→全角空白？の除去」
                                - 「日本語とか変な記号いっぱいあるから突っ込むべきかも。特に全角系。」
                          - text2phone
                            - 「本願寺。ここに`use_IPAg2p_phonemes`を引数として追加する」
                            - 「use_espeak_phonemesが使われていないけどこっちも使う予定ないし虫でいいや」
                          - intersperse
                            - 「vitsの、間に入れるやつ。これを、アクセントにも拡張しなきゃいけない」
                    - TTS/tts/utils/text/__init__.py
                      - pad_with_eos_bos
                        - 「enable_eos_bos_chars = True にすることで、音素の最初と最後に eos bos をつける。前後に無音があるなら使うべき」
              - TTSDataset.sort_items
                - 「日本語だけ特別扱いしていたが、それをやめる」
      - TTSDataset.collate_fn
        - TTSDataset.load_data
          - 「_load_or_generate_phoneme_sequence再登場。引数にaccentとuse_IPAg2p_phonemesを渡す」
          - 「sampleにaccent追加」
        - 「collate_fn内でもaccentを追加」
      - Trainer.train_step
        - Trainer.format_batch
          - BaseTTS.format_batch
            - 「ここで、アクセントも追加する」
        - Vits.train_step
          - 「acccentをforwardに渡す」
          - Vits.forward
            - 「テキストencoderにembeddingを足していく作業」
            - 「configは自動でaccent数とかを反映させたい」 
            - 「モデル改造周りはいつもどおりなので割愛」
          - Vits.forward_fine_tuning
            - 「同様」
    - Trainer.eval_epoch
      - 「中ではtrain_stepを使ったりしていて、つまり改造の必要はなさそう」
    - Trainer.test_run
      - Vits.test_run
        - Vits.get_aux_input_from_test_setences
          - 「アクセントを2番目で受け取るように設定」
        - TTS/tts/utils/synthesis.py
          - synthesis
            - 「これはtextにaccentを添えるだけだった」
            - 「なんかtfとかも回せるらしいけどtorch関連しか改造していません」
            - Vits.inference
              - 「すでに改造していたので、これもアクセントを添えるだけ」
## TODO
- TTS/tts/utils/text/cleaners.py: multilingual_cleaners
  - 日本語用に変な記号を除去するcleanerを実装しても良さそう。ワンちゃんg2IPAとかが対応しているかもだけど
- IPAg2pを行う際に`ver3`を決め打ちして使っているのでこれをconfigにかけるようにしたい
- phoneme_to_sequence のreturnにaccentを追加したので、それの反映が必要
- 補完について、実装的にアクセントに無駄tokenを埋められないので2倍に増やす形を撮ったが、もしかしたら無駄tokenのほうがいいかも