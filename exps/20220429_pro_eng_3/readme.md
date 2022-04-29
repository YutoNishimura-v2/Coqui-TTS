# 英語を話せる森川さん
- wandbの追加
- ちゃんと言語ごと，もしくは音声長で句切るチャレンジをやってみる

## wandb使い方まとめ
- あまり情報が転がっていない．
- 公式ドキュメントを全部読む．よさげな情報が落ちているので．

- quickstart:
  - 使うだけならマジで簡単．
  - login → init → logに突っ込む これだけ．
- Examples:
  - Audioは，logに投げるか，Tableとして投げるか．
    - tableとして投げるとメルとかと一緒に表示できるからいいかも?
  - tensorboard の結果をそのまま入力で切るっぽいけど，たぶん全ての実行が終わった後に載せることになりそうなので，逐次見たいときには使えないかも
- Guides:
  - Experiment Tracking:
    - initは，projectとconfigだけでとりあえずよさそう
      - tagsとかもいいけどね
      - projectにいつものexpname かな？
    - `wandb.watch(model)` これで勾配を見れる．やばい．やるべき 
      - 並列にしているときはmodel.moduleのほうがいいらしい
    -  Launch Experiments with wandb.init:
      - 1つのスクリプトで複数走らせる場合(abciとか)
        - abciじゃないかも？ここで言っているのは，マジで一回の実行で複数走らせること?
          - jobがわかれているなら話は別な気がする
        - run = wandb.init(reinit=True)
        - run.finish()
        - この2つが必要
      - multiprocessingするには
        - 詳細は別ページ見た方がよさそう
        - いろいろ方法があるみたいなので
    - Configure Experiments with wandb.config:
      - configに関しては完全に辞書．後からupdateするもよし．
      - 全訓練が終了した後も，apiだけ呼び出して後から追加とかできるぽい．便利
    - Log Data with wandb.log:
      - `wandb.log({'loss': 0.2}, step=step)` こんな感じでstep管理した方がよさそう
      - summary に重要な値を入れるべき．例えばbest_accuracy
      - デフォルトだと最後に追加した値がsummaryに入る
        - 最後というのは，最終更新．例: https://docs.wandb.ai/guides/track/public-api-guide
    - Send Alerts with wandb.alert:
      - pass
    - Dashboards:
      - pass
    - Limits & Performance:
      - いろいろな限界の話．まぁ最初は気にしなくてよさそう
    - Import & Export Data:
      - WANDB_API_KEY という環境変数にpassを入れればログインはできる
      - あとから値を取り出すとかもできる(上でやったあとからデータ追加と同じ)
    - Tracking Jupyter Notebooks:
      - pass
    - Advanced Features:
      - Distributed Training:
        - やり方は2つ
          - rank0 だけloggingする
          - 全てlogging する
        - 普通は1個めだけらしい．これはyourTTSに合わせる．
      - Resume Runs:
        - ひとまずinitにresume=Trueをつけておけばよさそう
  - Integrations:
    - PyTorch:
      - gradは絶対やる
      - profileができる！らしい
  - Collaborative:
    - Reports Walkthrough:
      - notionに張れるっぽい！！これはやるべき
  - Data + Model Versioning:
    - データのバージョン管理ができる．これも一行追加するだけなので簡単にできる．
    - https://wandb.ai/wandb_fc/japanese/reports/Weights-Biases---Vmlldzo4MDI5MTA
    - こんな感じっぽい．ちょっとよくわからないけど使いこなせたら便利そう


- 簡易版で試すべきこと:
  - resumeがうまく働くか
    - 正直管理方法がわからなかった．
    - nameとかではないので，同じシステムから稼働したら勝手に同じと認識する??
    - よくわからないけれど，再開毎に新しく作っても何の問題もないのでそうする．
  - multi-GPUでうまく動くか
    - TTS/trainer.py/Trainer/train_step
      - ここにおいて保存していたのを見ると，rank=0のものだけ．
      - 基本それでいいのだと思う．なので，そうする．

- 具体的に追加したところ
  - `run_at_abci.sh`
    - 環境変数に追加．
  - TTS/trainer.py/Trainer/__init__
    - 最下部でinitする．
    - 注意点としては，projectでそれ全体のイメージ．その下でexpとかを管理する感じ．なので，projectは大きく，「ひろゆき」とかそんな感じ
    - 再開するとき，configを変更するとエラーになる．普通再開したいときは変更しない気がするけど，そういう時は別runにするか上書きを許可するか．別のほうがよさそう．
    - yourTTSでは，毎回実行するたびにoutput_log_pathが更新されてしまうので，configに変更を食らう．なので，resume=Trueをナシにして，毎回再開するときでも別のnameでスタートすることにした．
      - たぶん，UIでは同時表示をできるはずなので管理に問題はないはず．
  - TTS/trainer.py/Trainer/train_step
    - ここで追加....
  
  - ここまでやって，「すでにwandbが実装されていた」ことに気づいてしまう．
  - なので，やるべきことは，project_nameとかを決めるだけ
  - configに書けばok.
```        
dashboard_logger = WandbLogger(
    project=project_name,
    name=config.run_name,
    config=config,
    entity=config.wandb_entity,
)
```

```
output_log_path_pathlib = Path(self.config.output_log_path)
_name = output_log_path_pathlib.parent.stem + "/" + output_log_path_pathlib.stem
wandb.init(
    name=_name,
    project=self.config.run_name,
    config=self.args,
)
wandb.config.update(self.config)
```

## 音声長で切るチャレンジ
- そもそも，引っかかるのは，vits.pyの
  `rand_segment(z, y_lengths, self.spec_segment_size)`
  ここ．
- そして，y_lengthsはmelの長さで，spec_segment_sizeはconfigにある．
- これで切ればいい．
- これを，TTSDataset.pyの`sort_items`で切る．
