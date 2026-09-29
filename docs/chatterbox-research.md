# Chatterbox Multilingual 与现有视频配音链路的兼容性调研

调研日期：2026-09-29。本文记录官方源码和项目当前源码能够证明的事实及实施要求。部署成功、运行速度和最终视频质量必须由实际部署与产物验收另行证明。

## 版本选择与可复现性

用户给出的 [Chatterbox-Multilingual-TTS Space](https://huggingface.co/spaces/ResembleAI/Chatterbox-Multilingual-TTS) 使用 **Multilingual V2**，不是英语版、Turbo、Nano，也不是后来发布的 Multilingual V3。以下版本来源分别固定：

| 来源 | 本次读取的 revision | 用途 |
| --- | --- | --- |
| 指定 Space | `c612a94210c0af895d6b649850d15b7ccca36594` | 确认用户指定实现及其默认生成参数 |
| 官方 Chatterbox GitHub | `5de7a54aa4e5e2baadb0182dde554908b48b85c2` | 使用包含 CPU 加载修复的运行代码 |
| `ResembleAI/chatterbox` 模型仓库 | `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18` | 固定实际模型、分词器和默认声线文件 |
| 官方 Perth GitHub | `ff1c8ac55a976971245cdd53c18d6131ca00d993` | 如采用当前源码版 Perth，固定水印实现及所含权重 |

上述 revision 分别通过 [Space API](https://huggingface.co/api/spaces/ResembleAI/Chatterbox-Multilingual-TTS)、[GitHub commit API](https://api.github.com/repos/resemble-ai/chatterbox/commits/5de7a54aa4e5e2baadb0182dde554908b48b85c2)、[模型 API](https://huggingface.co/api/models/ResembleAI/chatterbox/revision/5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18) 和 [Perth commit](https://github.com/resemble-ai/Perth/commit/ff1c8ac55a976971245cdd53c18d6131ca00d993) 核实。

当前官方说明推荐 V3，声称其说话人相似度与稳定性改善；这不等于本次指定 Space 已升级。V2/V3 均属约 500M 的多语言模型；Turbo 为 350M 英语模型，Nano 为 110M 英语模型。官方 Nano 的 CPU 性能宣称不能用于推断 Multilingual V2 的速度。[官方模型系列说明](https://github.com/resemble-ai/chatterbox/blob/5de7a54aa4e5e2baadb0182dde554908b48b85c2/README.md)

本次适配应明确加载 `t3_mtl23ls_v2.safetensors`。使用较新运行代码会包括上游修复，因此不能声称与旧 Space 逐样本相同。

## 代码与权重许可

Chatterbox 固定代码 `5de7a54…` 的 [LICENSE 为 MIT](https://github.com/resemble-ai/chatterbox/blob/5de7a54aa4e5e2baadb0182dde554908b48b85c2/LICENSE)；包含本次 V2 权重的 HF revision `5bb1f6e…` 的[模型卡也明确声明 `license: mit`](https://huggingface.co/ResembleAI/chatterbox/blob/5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18/README.md)。代码许可与权重声明分别核验，不能仅由代码的 MIT 推断权重许可。

本项目 Pocket TTS 固定代码 `adde0654…` 的 [LICENSE 为 MIT](https://github.com/kyutai-labs/pocket-tts/blob/adde0654090b1d54f6ee416ed10fd1b108069f13/LICENSE)，但其权重采用 **CC BY 4.0**：部署的 `--language english` 对应[固定代码中的 `english.yaml`](https://github.com/kyutai-labs/pocket-tts/blob/adde0654090b1d54f6ee416ed10fd1b108069f13/pocket_tts/config/english.yaml)，其中主权重指向 `kyutai/pocket-tts@39592ff23c9ef80098bb74895d104c26275fe2c9`，其[固定模型卡](https://huggingface.co/kyutai/pocket-tts/resolve/39592ff23c9ef80098bb74895d104c26275fe2c9/README.md)声明 `license: cc-by-4.0`；同一配置的无声音克隆后备仓库 `d29db797…` 的[固定模型元数据](https://huggingface.co/api/models/kyutai/pocket-tts-without-voice-cloning/revision/d29db7978e464fb90cb3359ee0c69a273b9142cc)也声明 `cc-by-4.0`。这是上游代码及权重的许可记录，不把模型许可扩展解释为参考声音素材或生成视频的统一许可。

## 官方接口与部署入口

以下是固定版本的核心加载 API 示意，避免 `from_pretrained()` 内部的 `revision="main"`。前提是先运行 `deploy/chatterbox/prefetch_models.py`，对六个文件分别核验完整文件大小和 SHA-256 后写入标准 Hugging Face 缓存；分片或未完成下载不能直接加载。`snapshot_download(..., local_files_only=True)` 只查找已预取的固定快照，不承担下载或完整性验证。

此片段不是独立部署脚本。生产镜像还必须应用并核验 [Space 对齐补丁](../deploy/chatterbox/apply_space_alignment.py)，加载后将整个 Llama 设为 eager；仅固定 V2 权重和参数不足以恢复 Space 的 EOS 行为。完整生产入口使用 [service.py](../deploy/chatterbox/service.py)，它还负责下文的 Cangjie/PKUSEG 离线加载、补丁哈希验证及请求隔离。

```python
from huggingface_hub import snapshot_download
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

checkpoint_dir = snapshot_download(
    repo_id="ResembleAI/chatterbox",
    revision="5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18",
    local_files_only=True,
    allow_patterns=[
        "ve.pt",
        "t3_mtl23ls_v2.safetensors",
        "s3gen.pt",
        "grapheme_mtl_merged_expanded_v1.json",
        "conds.pt",
        "Cangjie5_TC.json",
    ],
)
model = ChatterboxMultilingualTTS.from_local(
    checkpoint_dir, device="cpu", t3_model="v2"
)
model.t3.tfmr.set_attn_implementation("eager")  # 前提：镜像已应用并核验 Space 补丁。
reference_wav = None  # 模型 conds.pt 默认声音；克隆时改为已核验的参考 WAV。
wav = model.generate(
    text,
    language_id="en",
    audio_prompt_path=reference_wav,
    exaggeration=0.5,
    cfg_weight=0.5,
    temperature=0.8,
    repetition_penalty=2.0,
    min_p=0.05,
    top_p=1.0,
)
```

`from_pretrained(device="cpu", t3_model="v2")` 也是有效 API，但不固定下载 revision。模型原生返回 `[1, N]` 浮点张量，服务应明确编码成单声道、24,000 Hz、PCM16 WAV。当前官方代码支持 CPU 的 `map_location`，并删除末尾一个语音 token 对应的约 40 ms 噪声。旧 Space 的 `torch.load()` 缺少相应 CPU 映射，不能仅复制其 Gradio 界面就认定 ARM CPU 部署可用。[当前多语言实现](https://github.com/resemble-ai/chatterbox/blob/5de7a54aa4e5e2baadb0182dde554908b48b85c2/src/chatterbox/mtl_tts.py)、[采样率常量](https://github.com/resemble-ai/chatterbox/blob/5de7a54aa4e5e2baadb0182dde554908b48b85c2/src/chatterbox/models/s3gen/const.py)

当前客户端和服务显式使用上例六个生成参数，对齐指定 Space 的 V2 默认值，包括 `repetition_penalty=2.0`。当前 GitHub 同名函数默认值已改为 `1.2`。这是必须记录的行为差异，而不是可以忽略的依赖升级。协议中的 `speed=1.0` 仅限定自然语速，不传入模型的 `generate()`。[指定 Space 的多语言实现](https://huggingface.co/spaces/ResembleAI/Chatterbox-Multilingual-TTS/blob/c612a94210c0af895d6b649850d15b7ccca36594/src/chatterbox/mtl_tts.py)

参数相同不代表解码算法相同。部署真实样本后进一步发现：固定 GitHub 代码已经删除指定 Space 的对齐驱动 EOS 机制，具体差异见下节；因此此前候选构建不能视为已完成 Space 行为兼容。

另一个固定版本陷阱是中文分词初始化：`ChineseCangjieConverter` 仍调用未指定 revision 的 `hf_hub_download()`，即使顶层使用了 `from_local()`。部署需确保它读取已下载并核验的 `Cangjie5_TC.json`，不能在启动时静默读取新的 `main`；下载失败时上游仅记录 warning，可能导致中文处理降级。[分词器源码](https://github.com/resemble-ai/chatterbox/blob/5de7a54aa4e5e2baadb0182dde554908b48b85c2/src/chatterbox/models/tokenizers/tokenizer.py)

现有服务已将 Cangjie 下载调用绑定到固定快照的本地文件。分词器初始化还会加载 PKUSEG，即使本次输出为英语；部署先用 `prefetch_pkuseg.py` 预取并核验固定归档及解压文件，再由服务校验缓存并禁止隐式下载。仅设置 `HF_HUB_OFFLINE=1` 不替代这些检查或 `local_files_only=True`。

## 新发现：Space 的对齐驱动 EOS 与当前 GitHub 不同

2026-09-29 独立源码对照确认：Space `c612a94…` 的多语言 `T3.inference()` 创建 `AlignmentStreamAnalyzer`，每个采样步在 CFG 合并后调用它修改 logits；GitHub `5de7a54…` 删除了该模块的导入、构造及调用，并将两次模型前向的 `output_attentions=True` 改为 `False`。两版均使用自身采样循环，不调用 Hugging Face `.generate()`，因此不能通过补一个 `.generate(eos_token_id=...)` 参数恢复该机制。[Space T3](https://huggingface.co/spaces/ResembleAI/Chatterbox-Multilingual-TTS/blob/c612a94210c0af895d6b649850d15b7ccca36594/src/chatterbox/models/t3/t3.py)、[当前 GitHub T3](https://github.com/resemble-ai/chatterbox/blob/5de7a54aa4e5e2baadb0182dde554908b48b85c2/src/chatterbox/models/t3/t3.py)

恢复机制需保留以下具体语义；这些是源码条件，不是质量保证：

1. 分析器取零基层/头 `(12, 15)`、`(13, 11)`、`(9, 2)` 的注意力，只取 CFG 条件分支 `batch[0]`，三张注意力图等权平均。文本范围是 `[len_cond, len_cond + text_tokens.size(-1))`，不另行去掉文本 BOS/EOS。
2. 首次前向使用整个前缀，提取 `aligned_attn[j:, i:j]`；KV cache 后续每步只有一个 query，提取 `aligned_attn[:, i:j]`。分析器另执行 `A_chunk[:, curr_frame_pos + 1:] = 0` 限制对齐进度，这不是 Transformer 的因果 mask。原有条件、文本、语音/BOS 前缀拼接及 `speech_pos_emb(i + 1)` 不应在此次恢复中顺便改动。
3. 调用位于 CFG 合并之后、repetition penalty / temperature / min-p / top-p 之前；传入的是上一步 token，首步为 BOS。未到文本末尾（`cur_text_posn < S - 3` 且 `S > 5`）时抑制 EOS；检测到长尾、完成后回读或重复 token 时强制 EOS。实际重复条件为已记录至少三个 token 且最后两个相同，与源码注释“三连”不同；它不要求 `complete=True`。
4. 长尾与回读根据累计注意力判断，不是简单按文字长度设固定截止时间；强制 EOS 的索引必须为 `hp.stop_speech_token`。短文本 `S <= 5` 时，原回读检查的 `:-5` 区域为空；当前兼容实现对此增加非空区域条件，避免把空张量异常当作自然结束。

以上条件直接来自 [Space AlignmentStreamAnalyzer](https://huggingface.co/spaces/ResembleAI/Chatterbox-Multilingual-TTS/blob/c612a94210c0af895d6b649850d15b7ccca36594/src/chatterbox/models/t3/inference/alignment_stream_analyzer.py)。每次推理的分析状态应重新初始化，注意力 hooks 在成功、异常两条路径均须移除，避免连续请求积累旧状态。

Transformers 版本也是恢复条件的一部分：4.46.3 的 Llama SDPA 在要求注意力权重时回退 eager；5.2.0 的 SDPA 仅警告并返回 `None` 权重，单独设置 `output_attentions=True` 不足够。5.2.0 应在整个 Llama 前向之前选择 eager，让 `create_causal_mask()` 也生成正确的显式因果 mask；不能仅将三个被观察层换成 eager、却让上层仍按 SDPA 省略 mask。5.2.0 `LlamaAttention.forward()` 返回二元 `(attn_output, attn_weights)`，hook 应读取第二项并保持输出不变。KV cache 长度、`cache_position` 和 RoPE `position_ids` 保留模型原路径，不能因加入分析器而重置。[Transformers 4.46.3 Llama](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py)、[5.2.0 SDPA](https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/integrations/sdpa_attention.py)、[5.2.0 Llama](https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/models/llama/modeling_llama.py)、[5.2.0 mask](https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/masking_utils.py)

同文受控回归已验证该部署修复对一个真实长尾样本有效：相同 20 词原文、Jane 参考、seed 42 和六个生成参数，旧候选 `1a4cfe95…` 生成耗时 868.3809 秒、输出 32.44 秒，并因 2.78 秒内部停顿超过 0.80 秒阈值被拒绝；恢复 Space 对齐/EOS 并使用 eager 后，新候选 `c21819f5…` 的 job `b66f1ff459034c8999f50a9d8b9aa451` 生成耗时 267.0208 秒，输出 7.96 秒、24 kHz 单声道 PCM16。ASR advisory 报告 20/20 词匹配、首尾锚点完整、无警告；内部停顿检测为 0 段（报告阈值 0.5 秒），检查通过。[旧 job](../outputs/chatterbox-validation/20260929-065234-6fc5b7/deployment_attempts/v2-without-alignment/long-job.json)、[旧拒绝摘要](../outputs/chatterbox-validation/20260929-065234-6fc5b7/deployment_attempts/v2-without-alignment/long-integrity-summary.json)、[新音频及请求元数据](../outputs/chatterbox-validation/20260929-065234-6fc5b7/audio/tts_input_part_002_generated.json)、[新检查报告](../outputs/chatterbox-validation/20260929-065234-6fc5b7/audio/verification/tts_input_part_002/integrity_report.json)

这证明本部署的兼容修复消除了这个受控样本的长尾并通过既定检查；它不是对齐机制与 eager 两项变化的单独消融实验，也不能扩展为全模型稳定性、总体性能或主观听感保证。

后续完整 52 词样本已通过音频契约和物理检查：Chatterbox 四段无损拼接为 20.44 秒，ASR 精确匹配 52/52、无告警；同源 Pocket/Jane 三段为 20.28 秒，精确匹配 49/52、一个 advisory 告警，两方均接受完整 52 词分段覆盖。内部停顿检查均通过，最长分别为 0.56 / 0.69 秒。两者均为 24 kHz 单声道 PCM16、自然 1.0×；0.16 秒时长差不意味着波形或音色相同。[完整对照证据](../outputs/chatterbox-deployment-20260929/comparison.json)

已接受分段记录耗时合计分别为 833.844781 秒和 60.963 秒；Chatterbox 首段含新 worker 冷加载且使用服务端 job 计时，Pocket 使用含重试及响应规范化的客户端 HTTP 计时、冷热状态未单独记录，不能据此做热推理速度比较。最终 26.986667 秒、4K30 验收样片已通过现有质量、同步、混音、完整解码、HTTP 读回和 Ego Lite 连续播放；应用指纹已激活，原有默认选择保留。详细记录见[部署文档](chatterbox-deployment-20260929.md)。

## 与本项目现有 TTS 的差别

以下现有行为来自调研时的 `backend/config.py` 与 `backend/pipeline/tts.py`，不是历史文档推断。

| 维度 | Orpheus 项目适配 | Pocket TTS 项目适配 | Chatterbox Multilingual V2 需要的适配 |
| --- | --- | --- | --- |
| 运行形式 | 远程异步 job，轮询及下载 | 远程 `/tts` 流式 WAV | 独立远程异步服务，持久化 job 与请求去重 |
| 项目语言 | 英语 | 英语 | 模型和独立服务支持 23 种语言；当前项目视频适配器仅接受英语 `en` |
| 声线 | `leah` 等模型内置名称 | `jane`、`alba` 等服务声线 | 根据参考音频克隆；名称相同不代表模型天然共享声线 |
| 文本切分 | 默认最多 12 词的短片段及模型专用修正 | 以段落为主，默认 240 词；失败段落可改整句 | 按完整句子/标点有界切分，保留全部原文，不能照搬 240 词请求 |
| WAV | 检验 PCM 容器、帧数和时长 | 先修复流式占位 WAV 头，再做物理检查 | 服务输出完整 PCM16 WAV，避免把 float WAV 或不完整容器交给后续链路 |
| 文本识别 | 当前生成入口使用 ASR advisory | 当前生成入口使用 ASR advisory，并检查异常停顿 | 正式配音验证显式使用 ASR advisory，并保留物理音频/异常停顿检查；预览只做物理检查 |
| 拼接与节奏 | 无损 PCM 拼接，自然语速 | 无损 PCM 拼接，自然语速 | 同样保留采样、自然语速和以音频为准的视觉时间线 |
| 缓存 | 文本、音频和校验版本绑定 | 另绑定 provider revision 与 voice | 还需绑定模型权重、运行代码、参考音频哈希与生成参数 |

Chatterbox 的 `generate()` 会修改模型的 `self.conds`；参考声线和随机种子不能在同一模型实例上并行切换。服务需串行执行推理，外部可以排队。网络重试应追踪同一个已接受的 job；请求提交结果不明确时通过持久化幂等键找回，避免重复推理。

## 长文本、参考声线和自然语速

指定 Space 的 `app.py` 将文本直接截成 `text_input[:300]`；它是演示限制，直接接入会丢掉长稿后文。界面初始语言为法语，英语默认参考为 `en_f1.flac`，中文为 `zh_f2.flac`，清空上传后也会回到该语言参考。它们与项目现有的 `jane` 或 `leah` 无关。[指定 Space 的应用源码](https://huggingface.co/spaces/ResembleAI/Chatterbox-Multilingual-TTS/blob/c612a94210c0af895d6b649850d15b7ccca36594/app.py)

模型一次推理固定最多生成 1,000 个语音 token，语音编码器速率为每秒 25 token，因而单次输出上限约为 40 秒，而不是任意长度朗读。2,048 个文本 token 的配置上限不等于 2,048 token 长稿都能完整发声。适配层应按句切分，限制字符和词数，遇到过长单句继续在标点或词边界切分，拒绝静默截断；需要同时用边界样本验证开头、结尾及跨片段连接。[语音 token 常量](https://github.com/resemble-ai/chatterbox/blob/5de7a54aa4e5e2baadb0182dde554908b48b85c2/src/chatterbox/models/s3tokenizer/s3tokenizer.py)、[T3 配置](https://github.com/resemble-ai/chatterbox/blob/5de7a54aa4e5e2baadb0182dde554908b48b85c2/src/chatterbox/models/t3/modules/t3_config.py)

模型支持的语言代码为 `ar da de el en es fi fr he hi it ja ko ms nl no pl pt ru sv sw tr zh`。参考音频应与输出语言匹配。沿用 `jane` 时，应将已调好的现有旁白中干净、单人、无背景音乐的片段登记为独立克隆参考，记录来源和 SHA-256；对比使用同一段文字。不能把官方默认参考标成现有 `jane`。参考编码分别使用前 6 秒和前 10 秒；整段语音还用于说话人嵌入。

当前独立服务的 `default` 指固定模型自带的 `conds.pt`，与 Space 的英语 `en_f1.flac` 是不同参考；服务只在配置的 `jane` 参考 WAV 存在时暴露该声线。每次推理前恢复模型原始条件，避免一次 `jane` 克隆污染后续 `default` 请求。客户端按服务返回的参考指纹核验声音身份，不因名称相同就接受其他声音。

模型没有原生 `speed` 参数。`cfg_weight`、`exaggeration` 可以影响韵律和主观速度，但没有与 1.0×/1.2× 对应的时间比例契约。固定种子有利于同环境复现，不保证不同硬件、依赖版本或模型修订之间逐样本一致。不得通过 FFmpeg `atempo` 强行让新模型配音匹配旧视频时长。

## ARM CPU 与依赖

当前 Chatterbox 官方项目元数据要求 Python ≥3.10；Python 3.11 对应 Torch/Torchaudio 2.6.0、Transformers 5.2.0、Librosa 0.11.0、Diffusers 0.29.0、Safetensors 0.5.3、NumPy 1.x。其 GitHub 元数据把 Perth 指向 mutable `master`，应单独固定。原 Space 使用 Transformers 4.46.3、Librosa 0.10.0 和 Perth 1.0.1，不能混合两套锁定版本后假定兼容。[当前 pyproject](https://github.com/resemble-ai/chatterbox/blob/5de7a54aa4e5e2baadb0182dde554908b48b85c2/pyproject.toml)、[Space requirements](https://huggingface.co/spaces/ResembleAI/Chatterbox-Multilingual-TTS/blob/c612a94210c0af895d6b649850d15b7ccca36594/requirements.txt)

PyPI 提供 Python 3.11 的 Torch/Torchaudio 2.6.0 `manylinux_2_28_aarch64` wheel，以及 `spacy-pkuseg` 1.0.1 的 aarch64 wheel。这证明存在对应架构发行物，不证明目标机器依赖、glibc、内存和推理速度已经合格。部署应使用隔离环境，并先验证 `uname -m`、glibc、import、模型加载、真实请求、耗时及峰值内存。[Torch 发行物](https://pypi.org/pypi/torch/2.6.0/json)、[Torchaudio 发行物](https://pypi.org/pypi/torchaudio/2.6.0/json)、[中文分词发行物](https://pypi.org/pypi/spacy-pkuseg/1.0.1/json)

当前 Perth 官方源码采用 PyTorch 实现，默认在 CPU 加载，随仓库包含约 37.4 MB 的水印权重。应确认实际安装保留权重，并完成水印实例加载；仅有 `import perth` 成功不够，因为顶层可把失败的 `PerthImplicitWatermarker` 设置为 `None`。Chatterbox 默认生成中调用水印，不应通过无声替换为 dummy 的方式伪装兼容。[Perth 实现](https://github.com/resemble-ai/Perth/blob/ff1c8ac55a976971245cdd53c18d6131ca00d993/src/perth/perth_net/perth_net_implicit/perth_watermarker.py)、[Perth 顶层导入](https://github.com/resemble-ai/Perth/blob/ff1c8ac55a976971245cdd53c18d6131ca00d993/src/perth/__init__.py)

另已直接检查 PyPI Perth 1.0.1 wheel：它也是 Python 源码加预训练权重，没有平台专用动态库，但其 `perth/perth_net/__init__.py` 依赖 `pkg_resources`。需要验证当前 setuptools 是否仍提供该模块，或采用已经移除此依赖的固定官方源码版本。[Perth 1.0.1 发行物](https://pypi.org/pypi/resemble-perth/1.0.1/json)

## 视频一致性及验收边界

更换模型后，音色、停顿、语气和实际时长可能变化，不能承诺波形或旧视频逐帧相同。应保持的是已有视频流程的制作约束和验收标准：完整稿件、正确选用的参考声线、自然速度、真实音频时长驱动画面、完整字幕与镜头顺序、无损中间 WAV、可追溯的音频和文本哈希。

1. 对同一项目现有脚本分别生成基线和新配音，记录模型/代码/权重/声线参考/参数、合成时长和音频时长。基线必须来自当前已调好的实际 provider 与 voice。
2. 验证至少一段超过 300 字符的稿件、多句边界、数字和技术专名；输出原文与分块拼接文本必须完全保真。中文输出如纳入项目产品能力，需要另外验证语言对应的 ASR 与字幕，不因模型支持中文就声称下游完成中文适配。
3. 保留现有 WAV 完整性、时长、异常停顿检查及 ASR 参考报告。当前 TTS 配音验证使用 `asr_advisory`；ASR 未识别到或拼写不同不会单独触发配音重试，也不能据此省略人工对实际音频的判断。后续视频同步和质量验收仍按现有流程独立执行。
4. 缓存身份必须区分旧 provider、新 provider、语言、声音参考哈希、运行修订及采样参数。中断恢复复用同一身份的完整片段。
5. 通过项目现有合成路径产出最终视频，检查最终 artifact 状态、音视频同步/质量报告、FFprobe 与完整解码、HTTP 端点以及 Ego Lite 的实际连续播放。服务 `/health` 或单个 WAV 成功不是视频验收完成。

源码/API/发行物事实、本次 52 词音频对照、40 词专名数字补测及完整 4K30 样片验收置信度高；受控冷热推理速度、长稿压力稳定性、主观声线盲评与其他语言的视频认证未完成，不能将这些样本推广为所有文本的质量保证。最终证据及真实审核 provider 见[部署记录](chatterbox-deployment-20260929.md#最终视频验收)。
