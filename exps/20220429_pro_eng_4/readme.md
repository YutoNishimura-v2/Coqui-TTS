# log問題解決のための試行錯誤
- まずは，簡単ケースから攻めていく

1. インタラクティブ+1GPU
- コンソールに出力はあるものの，`trainer_0_log.txt` には出力は出ず
  - つまり，大元の問題の可能性が．

## printについて
- printは実は，`sys.stdout.write`を行っているだけ．
- そして，`sys.stdout`はオブジェクトであって，これを上書きしてしまえば，printした時に自分で作ったクラスから発動されるということ．
- それを用いて，yourTTSでは sys.stdout = Loggerとしている．
- 常にopenになってると，書き終わりは実行終わり．これのせいな気がする
- なので，あほみたいな実装ではあるが，withでいちいち開いて記述にした
  - 成功．できた.

- これだと，本来の標準出力には出さない．これのせいで，wandbには出さない
- なので，本来の標準出力にも出すようにする．
- すでにterminalに標準出力はだしていた
- `-o`にしているのが悪いのでは??
  - ダメでした．
- batch job 側の問題では??
  - `-o` に wandb の output.log を指定してみたけどダメだった．

- train_tts.py のmain() の直後にprintしてみたら，`-o` で指定したlogには出現したけど，wandbには出てこず．ついでに言うと，wandbの出してくるメッセージもwandbに反映されず．これは，インタラクティブと同様

- 常に(wandbの放つメッセージ以外) `-o` とwandbのLogsの内容は一致しているから，出力先の問題の可能性？
  - terminalに明示的にsys.__stdout__にしてもダメだった...．

- 出力がどこかの段階で変わっている．
  - main() 直後だとprintできるのに，_setup_logger_configの中だと，Logger初期化前でもアウト．どこかで悪さしてるやつがいそう
  - いろんな場所でprintしてみておかしくなった瞬間を調べる
    - print("h")まで
    - print("ee")まで 
    - print("jjj")なで
    - print("llll")なで
  - WandbLogger ここを通った瞬間にダメになっていた...
    - init した瞬間にダメになる...なぜ...
    - wandbのソース見たら，中でloggerを作ってた．これのせいか?
    - ただ，tqdmはこれを貫通してくることに注意したい．これをみればいいのでは？
    - tqdmのソースコードはよくわからないが...いい記述を発見
      - https://stackoverflow.com/questions/36986929/redirect-print-command-in-python-script-through-tqdm-write
    - tqdmはデフォルトでエラーに出力しているらしい?!
    - `sys.stdout = sys.stderr` とかは意味なし....
    - やっぱりwandbのinit側がかなり悪いことしてる気がする

- 一旦 tensorboard を実行してみる
  - お　な　じ
  - tensorboard にしても，`-o` には出てこなかった
  - wandb特有の問題ではないということ
  - tensorboardにしたら，別の位置からprintできなくなった．
    - init_dashboard_logger の前からprintしても無理．意味不明．
    - その代わりtqdmはしっかり表示される
  - main()の直後からもはや出力が無理になってしまった...
    - main()の直前に書いても無駄
  - さすがにpythonの前にechoしたらできたが．．．
  - 読み込み始めるのがdashboradのところが初めてなはずなのに本当におかしい

- 適当な `    "dashboard_logger": "nkjn",` とかにしたら？
  - printはできた．やはり tensorboardにだけ反応している....
  - "tensorboad" でもできた

- さすがに意味不明なので諦める．
  - train_0_log.txt には一応出てくるからね...
  - 一応最後に先生に聞く

- 状況:
  - abciのバッチジョブに入れたときだけ標準出力が正常に作動しない
  - 具体的には，「configにwandbを指定するとinitの直前まではprintが起動する」
  - 「configにtensorboardを指定するとmain()の直前からもうprintが起動しない」
  - configの内容を見ているはずがないのに反応するのがやばい．意味不明．

- 後可能性があるのは import しているときに何かを読み込んでいないか
  - ipdb しても量が多すぎてよくわからない
  - import してるモジュールは全部見てみたけど悪いことはしていなさそう...
  - debugはさすがにvscodeのやつのほうが何倍も見やすい
  - それを見た感じ，事前に実行しているコードはいくつかあった
    - 音素の変換準備とか，
    - @dataclass 系とか
    - 全ての関数を一応読み込んでいる
    - でも，何かtensorboardとかに絡む読み込みはなかった気がする...

- import でも print作戦してみる
  - 