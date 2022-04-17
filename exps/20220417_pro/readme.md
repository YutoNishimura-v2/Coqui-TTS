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
