# ComfyUI MiniMax H3 PDMD X2

日本語 | [English](README.md)

MiniMax H3のFL2VAを、**事前融合したPDMD 2step・SLA keep 20%・X2 VAE INT8ストリーミング**で生成するためのパッケージです。サンプルは**544×960でサンプリングし、1088×1920・124フレーム・24fps**の音声付き動画を保存します。

インストール・CPUでの融合モデル作成ツールと、ComfyUIへ読み込めるワークフローを収録しています。このフォルダは `ComfyUI/custom_nodes` の外に置いてください。setupが追加ノード [ComfyUI-H3-X2-Stream](https://github.com/sepiablue-ai/ComfyUI-H3-X2-Stream) を対象のComfyUIへ導入します。

[Sayaka Benchmark](https://sayakabenchmark.pages.dev/)での対応番号は**FL2VA 015**です。**既知の制約：この2step設定では生成音声の品質が悪くなることがあります。**

## 必要な環境

- Windowsと、MiniMax H3・INT8 ConvRotに対応した既存のComfyUI。検証したcoreは0.38.0、commit `6b747c0428c343e1417219641db93a4fb7cb69ae`です。融合時は関連ソースのハッシュを検査し、互換性を確認できない実装では停止します。
- そのComfyUIのvenvまたはWindows Portable付属Python。新規のWindows clone・venvでPython 3.13.13、PyTorch 2.14.0+cu130、torchvision 0.29.0+cu130、comfy-kitchen 0.2.36、safetensors 0.8.0、psutil 7.2.2、PyAV 19.0.1、transformers 5.18.0を使って検証しています。
- PATH上のGit。X2ノードのフォルダがなければ、setupがcommit `eb148a3d0f91a273668bbe027bfcbafb85f2c5e8`を導入します。既存のノードは保持します。パッケージ導入には、利用できればuv、なければ指定Pythonのpipを使います。
- 動画生成用のNVIDIA CUDA GPUとH.264 NVENC。RTX 4070 12 GBで検証しています。
- 元モデルとは別に、融合先ドライブの**空き容量30 GB以上**。融合済みモデルは約21 GBです。融合はCPU・2スレッドで行い、システムの空きRAMが8 GiB未満になると待機します。

setupはComfyUI本体とX2ノードのrequirementsを、指定したComfyUI用Pythonへ導入します。CUDA版PyTorchとtorchvisionは事前に用意してください。導入済みtorch・torchvision・torchaudioの版は保持します。モデルの重みは別途用意します。ComfyUIのcore変更とサーバー起動は行いません。

新規のWindows環境では、対応するcoreへ固定し、venvを作ってからsetupを実行します。Gitとuvを使用する具体例は[英語README](README.md#requirements)にあります。

## モデルの準備

各モデルの利用条件に従って、[英語READMEのモデル一覧](README.md#models)にあるファイルを用意してください。モデルの重みは同梱していません。

X2用には、元の `MiniMax-H3-X2-Detail-v1.safetensors` を `models/vae/` に置きます。付属サンプルの「H3 X2 Prepare INT8 VAE」で元モデルを選ぶと、初回キュー時に `models/vae/h3_x2_stream/MiniMax-H3-X2-Detail-v1-decoder-int8-convrot.safetensors` を作成します。次回から変換済みファイルを再利用します。

## PDMD融合モデルの作成

1. `config.example.json` を `config.local.json` にコピーします。
2. `comfyui_root` に `main.py` と `comfy/` があるフォルダ、`base_model` と `pdmd_lora` に元モデルのパス、`output_directory` に融合先を設定します。相対パスは設定ファイルのフォルダが基準です。`fusion_manifest` はnullのままで、出力モデルと同じ場所に記録されます。
3. ComfyUI用Pythonを指定して実行します。venvの場合の例です。

```bat
setup.bat "C:\ComfyUI\.venv\Scripts\python.exe" "config.local.json"
setup.bat "C:\ComfyUI\.venv\Scripts\python.exe" "config.local.json" --check
```

Windows PortableではPythonのパスを、その環境の `python_embeded\python.exe` に置き換えてください。システムPythonは使わないでください。

通常のsetupは依存関係を導入した後、検証済みの融合モデルがあれば再利用し、なければCPUで作成します。出力名は `minimax_h3_pdmd_2nfe_fused_int8_convrot.safetensors` です。200個のINT8 ConvRotモジュールと8個のBF16モジュールを融合し、変更箇所と未変更データを検証します。元のbase・LoRAは変更しません。無関係な既存ファイルや作成途中の出力がある場合は停止します。`--check` は導入済み環境に対する書き込みなしの検査で、パッケージやノードの導入は行いません。

4. 設定ファイルと同じ場所にsetupが作る `extra_model_paths.local.yaml` のセクションをComfyUIの `extra_model_paths.yaml` に追加するか、起動オプション `--extra-model-paths-config` で指定します。融合先がComfyUIの `models/diffusion_models` であれば自動で見つかります。通常のランチャーでComfyUIを起動してください。

融合のmanifestと進捗ファイルは、設定したモデル出力先に保存されます。融合と動画生成は別の操作です。

## ComfyUIでの生成

1. setupが[examples/assets](examples/assets)のPNGを `ComfyUI/input/` へコピーします。同じ内容の画像は保持し、同名で異なるファイルは上書きせず停止します。
2. [case01](workflows/fl2va_015_case01.json) または [case02](workflows/fl2va_015_case02.json) をComfyUIへドラッグします。[API形式](workflows/api)もあります。
3. 融合済みPDMDモデル、text encoder、通常のH3 video VAE、audio VAE、元のX2 Detail v1モデルを選びます。
4. 保存ノードのNVENC GPU番号を自分の環境に合わせ、キューへ送ります。

**融合済みモデルへPDMD LoRAを再適用しないでください。** 通常のH3 video VAEは画像の条件付けに、X2 VAEは最後の動画デコードに使います。

設定はEuler/simple、2step、BasicGuider、seed 43、video/audio shift 12/3、SLA keep 20%、FFN chunks 8、NVENC CQ 18、`fp16_accumulation=true`です。このPyTorch機能を利用できない場合は最後のオプションを無効にしてください。動画はComfyUIの出力先の `FL2VA_015_PDMD_X2/` 以下に保存されます。

## 困ったとき

- モデルや実装のハッシュ不一致：`model_recipe.json` の元モデルと対応実装を使用してください。
- X2ノードがない：ComfyUI用Pythonでsetupを実行してから再起動してください。X2 INT8 decoderはサンプルの準備ノードが初回キューで作成します。
- 融合モデルが選択肢にない：モデル探索設定を確認し、ComfyUIを再起動してください。
- 融合が中断した：出力先を確認してから再実行してください。setupは部分ファイルや既存manifestを上書きしません。
- 音質が悪い：この2step設定の既知の制約です。CPUでの融合・再量子化により、LoRAを動的に適用した場合と動画・音声が変わることもあります。

既存モデルの完全なCPU検証は、ComfyUI用Pythonで `-B scripts/fuse_pdmd_cpu.py --config config.local.json --verify` を実行してください。再融合は行いません。

## ライセンス

コードとワークフローの変更部分はGPL-3.0です。[LICENSE](LICENSE)と[NOTICE.md](NOTICE.md)を参照してください。モデルの重みには、それぞれのモデルの利用条件が適用されます。
