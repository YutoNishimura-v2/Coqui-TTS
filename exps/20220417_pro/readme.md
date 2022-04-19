# pro多話者詰め合わせ
- proのみで多話者TTSは可能なのかの実験
- @dataclass によって作られていると、型を自動で変えられてしまい、Unionとかも無理
    - なので、unused_speakerにはリストで渡す

- multi-GPUは前途多難...
```
Traceback (most recent call last):
  File "/home/acd14006vc/venv/yourTTS/bin/kernprof", line 8, in <module>
    sys.exit(main())
  File "/home/acd14006vc/venv/yourTTS/lib/python3.8/site-packages/kernprof.py", line 220, in main
    execfile(script_file, ns, ns)
  File "/home/acd14006vc/venv/yourTTS/lib/python3.8/site-packages/kernprof.py", line 28, in execfile
    exec(compile(f.read(), filename, 'exec'), globals, locals)
  File "TTS/bin/train_tts.py", line 15, in <module>
    main()
  File "TTS/bin/train_tts.py", line 10, in main
    trainer = Trainer(args, config, output_path, c_logger, dashboard_logger, cudnn_benchmark=False)
  File "/home/acd14006vc/venv/yourTTS/lib/python3.8/site-packages/line_profiler/line_profiler.py", line 110, in wrapper
    result = func(*args, **kwds)
  File "/home/acd14006vc/gcd50804/yuto_nishimura/workspace/python/yellston/TTS/TTS/trainer.py", line 262, in __init__
    init_distributed(
  File "/home/acd14006vc/gcd50804/yuto_nishimura/workspace/python/yellston/TTS/TTS/utils/distribute.py", line 20, in init_distributed
    dist.init_process_group(dist_backend, init_method=dist_url, world_size=num_gpus, rank=rank, group_name=group_name)
  File "/home/acd14006vc/venv/yourTTS/lib/python3.8/site-packages/torch/distributed/distributed_c10d.py", line 627, in init_process_group
    _store_based_barrier(rank, store, timeout)
  File "/home/acd14006vc/venv/yourTTS/lib/python3.8/site-packages/torch/distributed/distributed_c10d.py", line 255, in _store_based_barrier
    raise RuntimeError(
RuntimeError: Timed out initializing process group in store based barrier on rank: 0, for key: store_based_barrier_key:1 (world_size=8, worker_count=1, timeout=0:30:00)
```

- まず、実行のやり方から違う。これは Coquitの https://github.com/coqui-ai/TTS/blob/main/docs/source/faq.md ここに。
  - 普通にやってしまうと動かなくなる (ctrl+C が動かない)
- addressは、一度使うと再起動しないとだめらしいので、config からアドレスをいちいち変えて実験する必要あり

- log が作れていないのが気になったので、"./"をログパスにつけてみた
  - うまく動いて、しかも今まで進まなかったところから進むようになった。
  - 一方で、別のエラー: `FileNotFoundError: [Errno 2] No such file or directory: 'exps/tts_models--multilingual--multi-dataset--your_tts/model_file.pth.tar'`
  - これ、outputの件もそうだけど、パスの指定が悪い気がしてきた。ちゃんと相対パスで書くことにする。
  - 相対パスで書いても無駄だった。
  - しかも、一番上のをもう一回試したけどだめ。再現性なし

- ↑多分解決。想像舌原因: multi-processなのにopenをしているだけで、withがない。
  - つまりなんか同時に開いてしまったりしてopenはそれはできないのではないかという仮設
  - 実際、sleepを挟んであげたら何回試してもうまく動いた。やった。

- 動いた。本当にここだけが問題だった。

- batchサイズに関しては、現状 52 だけど、https://aru47.hatenablog.com/entry/2020/11/06/225052 ここにあることを考えるならGPUに比例してbatchは増やす上に、lrもちょっといじる必要があるそう。調整しなきゃ？

- ↑一度実験的にやってみる。8倍までやるのと、シングルとの実行速度比較。
- 以下はpro+engの方。
