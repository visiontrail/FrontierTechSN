# Chatterbox Multilingual V2 部署运行记录：2026-09-29

最终状态（2026-09-29 09:09 UTC）：Chatterbox Multilingual V2 已部署到 nr-test，指纹 `c21819f5…` 已通过应用设置接口激活。52 词、约 26.99 秒的 4K30 验收样片通过现有制作、音视频同步、混音、多模态审核、完整解码、HTTP 读回和 Ego Lite 连续播放检查。原有 Pocket/Jane Morning Desk 配置及全局默认模型保留。这里确认的是本次样片和制作契约一致，不承诺不同模型的音色、波形或所有长稿结果完全相同。

相关文档：[模型及兼容性调研](chatterbox-research.md)、[完整视频验收流程](chatterbox-validation.md)、[Chatterbox 部署 runbook](../deploy/chatterbox/README.md)、[Pocket 部署 runbook](../deploy/pocket-tts/README.md)。运行证据保存在 ignored `outputs/`，以下链接为本工作区留存路径，不表示这些产物已进入 Git。

## 主机与服务隔离

现场核对的 nr-test（`10.60.11.3`）为 ARM64、Ubuntu 18.04、Linux 4.15，64 个逻辑 CPU、约 124 GiB 内存，无 GPU。新服务使用 Python 3.11 / Debian Bookworm 容器提供所需运行库，配置 CPU 推理、8 个 Torch/BLAS 线程。

| 服务 | 地址 | 本次状态 |
| --- | --- | --- |
| Chatterbox Multilingual V2 | `10.60.11.3:8092` | 独立容器、缓存、任务状态目录和参考声音 |
| Pocket TTS | `10.60.11.3:8090` | 原服务保留，未迁移到新端口 |
| Orpheus | `10.60.11.3:8088` | 原服务保留，未迁移到新端口 |

Chatterbox 部署目录为 `/home/guoliang/chatterbox-tts-deploy`。`/health` 只证明 HTTP 服务就绪，模型在首个任务加载；模型就绪必须另用真实合成证明。API 密钥保存在运行配置中，本记录不包含密钥或会话凭据。

## 已验收构建身份

以下是加入固定 Space 对齐机制、eager 注意力及 hooks 生命周期处理后，实际通过本次完整验收并被应用接受的构建。来源为[远端操作记录](../outputs/chatterbox-deployment-20260929/ssh-terminal.txt)、[能力响应](../outputs/chatterbox-deployment-20260929/capabilities.json)及[完整依赖列表](../outputs/chatterbox-deployment-20260929/runtime-requirements.txt)。

| 对象 | 固定版本 / SHA-256 |
| --- | --- |
| Docker image ID | `sha256:993b6870c608a9f729cd8c5bec5e614cb09d35b3caddbac39341764f5fabd007` |
| Chatterbox 源码 commit | `5de7a54aa4e5e2baadb0182dde554908b48b85c2` |
| HF 模型 snapshot | `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18` |
| 模型变体 | `v2` / `t3_mtl23ls_v2.safetensors` |
| Perth 源码 commit | `ff1c8ac55a976971245cdd53c18d6131ca00d993` |
| Space 算法来源 revision | `c612a94210c0af895d6b649850d15b7ccca36594` |
| 注意力实现 | `eager` |
| `space_alignment.py` SHA-256 | `380d58d3e8decf1dcc9d32fa194f1ce576fce82ce8b34731fba32b290ebcbee7` |
| `apply_space_alignment.py` SHA-256 | `625a1865fa04daba2a2cc45b4b79e107c1000c687984953fe1533b7e79f03c20` |
| 对齐补丁 manifest SHA-256 | `f5af62ef5bc66d786994394d3023dbd700479674d3f58cd1da75b440f6cf0d67` |
| `service.py` SHA-256 | `faa8298e3d5f3a5632697c19c07f988c1acdec9413c46945a90d06adec1a309d` |
| 能力 / 部署指纹 | `c21819f5d3081b91f1e641e50bd2ea09f24f9a373ae98875ca9127abda65d436` |
| `runtime-requirements.txt` SHA-256 | `ec32565be53bbdfce69132dbe173ebc78efaedeae5366b9e11b542c6f718ad83` |
| PKUSEG loader SHA-256 | `4dd95958d507599e7529849576b0d22010fc6ef271bd2c8a939bef08a4dcc641` |

输出契约为 24 kHz、单声道、PCM16 WAV，自然语速 `speed=1.0`。生成参数为 `exaggeration=0.5`、`cfg_weight=0.5`、`temperature=0.8`、`repetition_penalty=2.0`、`min_p=0.05`、`top_p=1.0`；水印使用固定版本的上游 Perth。接近 V2 解码上限的输出（时长 ≥38.8 秒）被拒绝，不作为完成音频发布。

补丁以原始 T3 文件 SHA-256 核验替换位置，并将分析器、补丁器与补丁 manifest 哈希纳入能力指纹。目标 ARM64 容器的 Torch 2.6.0 / Transformers 5.2.0 已运行最小真实 Llama smoke：显式 eager 因果 mask、三组注意力 hook、前缀及单步 KV cache、两次调用后的 hook 清理通过。目标机 `alignment-build.log` 的构建步骤 `#13` 记录 `Fixed Space alignment patch verified` 和 `Space alignment eager/cache/hook smoke passed`，`alignment-build.exit` 为 `0`；该 smoke 不加载完整 TTS 权重，也不等同于真实配音质量通过。[构建日志](../outputs/chatterbox-deployment-20260929/alignment-build.log)、[补丁器](../deploy/chatterbox/apply_space_alignment.py)、[分析器与 smoke](../deploy/chatterbox/space_alignment.py)

## 参考声音与离线资源

`jane` 取自已有成功 Pocket/Jane 任务 `20260928-235341-4aa80a` 的第一段干净旁白：[原始参考 WAV](../outputs/20260928-235341-4aa80a/audio/tts_input_part_001_generated.wav)。该文件为 24 kHz、单声道、PCM16、10.92 秒，SHA-256 为 `4c3f950e7e549b41143921da5e5d8e477573d4104d74b03f70fcdeb014a10060`。来源与基线元数据见 [baseline.json](../outputs/chatterbox-deployment-20260929/baseline.json)。这是基于该旁白的参考克隆，不代表两个模型天然共享同一个内置声音。

`default` 使用固定模型的 `conds.pt`，其 SHA-256 为 `6552d70568833628ba019c6b03459e77fe71ca197d5c560cef9411bee9d87f4e`；它不是 Space 的 `en_f1.flac`。两种声音分别具有能力响应中的参考指纹，参考文件变化会使旧身份失效。

六个模型/分词器文件合计 3,208,951,748 bytes，已按固定 snapshot 预取；完整文件大小和 SHA-256 匹配后才进入标准 HF 缓存。固定清单与验证逻辑见 [prefetch_models.py](../deploy/chatterbox/prefetch_models.py)。Range 分片和中断残留只作为续传候选，不能替代完整模型哈希验证。

实测仅设置 `HF_HUB_OFFLINE=1` 时，安装的 Hugging Face Hub 1.33.0 仍尝试查询仓库树；服务因此显式调用 `snapshot_download(..., local_files_only=True)`。Cangjie 映射也固定到本地 snapshot。PKUSEG 会在英语模型初始化时加载，需另外预取 `spacy_ontonotes.zip`（34,567,143 bytes，SHA-256 `b216e7f92de7ae285aeab8feba2faa8ea8216e5995ff6fb3d391cc8356db1bfe`），校验归档及解压文件，并禁用请求期间的隐式下载。具体入口见 [service.py](../deploy/chatterbox/service.py) 和 [prefetch_pkuseg.py](../deploy/chatterbox/prefetch_pkuseg.py)。

网络较慢时使用已核验的 Linux ARM64 / CPython 3.11 wheelhouse 完成构建；96 个 wheel 包含运行与隔离构建依赖，离线依赖解析、平台标签、发行物 SHA 和二进制架构审计通过：[wheel 索引](../outputs/chatterbox-deployment-20260929/wheelhouse/index.html)、[发行物审计](../outputs/chatterbox-deployment-20260929/wheelhouse-artifact-audit.json)、[离线 pip 检查](../outputs/chatterbox-deployment-20260929/wheelhouse-offline-check.log)。Perth 归档由固定官方 Git tree 重建，37 个文件均通过 Git blob SHA-1 校验，归档再经 SHA-256 和读回复核：[来源清单](../outputs/chatterbox-deployment-20260929/perth-verified-source-manifest.json)、[归档读回复核](../outputs/chatterbox-deployment-20260929/perth-archive-readback-audit.json)。

## 已完成的检查及其边界

针对本次集成的最新检查记录为 **403 tests passed、171 subtests passed**，另有一个现有 `twitter_text` 依赖的 `pkg_resources` 弃用警告；这些是代码/契约测试，不是最终视频验收。[测试日志](../outputs/chatterbox-deployment-20260929/final-contract-tests.log)

真实 HTTP 检查确认超长文本、非自然语速、未知声音、未知语言返回 422；该轮负向检查没有提交模型任务。[HTTP 检查记录](../outputs/chatterbox-deployment-20260929/http-validation.json)

应用模型注册表已出现 `chatterbox-multilingual`（`Multilingual V2 (English delivery)`），声音为 `jane` 和 `default`；Ego Lite 中也已现场确认该模型选项存在。项目视频入口目前只接受英语，不能将服务支持 23 种语言等同于项目已完成 23 种语言的字幕/视频验收。原有默认模型设置未改，API 注册表仍将 `vibevoice-0.5b` 标记为默认，新模型 `is_default=false`。[应用接口记录](../outputs/chatterbox-deployment-20260929/app-integration.json)

排障期间曾清空 `CHATTERBOX_TTS_DEPLOYMENT_FINGERPRINT`，仅允许独立验证进程调用候选。完整验收通过后，已在 09:09 UTC 通过 `/api/settings/values` 保存并在线生效 `c21819f5…`，无须再次重启。应用 `/api/voices/jane/preview?tts_model=chatterbox-multilingual` 返回 200，音频 SHA 与正常预览生成器缓存一致。Morning Desk 完整设置哈希前后相同（Pocket/Jane、自动化关闭、`auto_publish=false`），全局默认仍为 `vibevoice-0.5b`。[激活及读回证据](../outputs/chatterbox-deployment-20260929/application-activation.json)、[试听生成](../outputs/chatterbox-deployment-20260929/preview-generation.json)

验证脚本新增 `--deployment-fingerprint`，仅允许 Chatterbox 的 `run` 模式接受完整小写 SHA-256，并只覆盖该验证进程中的配置变量，不写保存的应用设置。本次独立验收先使用此入口绑定 `c21819f5…`，通过后才按上文激活保存的应用设置。[验证 CLI](../scripts/validate_chatterbox.py)

本次验收期间，`FrontierTechSN 视频巡检与连续生成` 自动化保持 `PAUSED`；记录时已读取对应本机 `automation.toml` 确认。此状态不等于关闭应用任务 worker，也不免除验收脚本开始前的生产任务空闲检查。

## 首次真实成功样本

首次完成的 job 为 `e4f7cf4612a8478393e639bcccb4055e`，使用 Jane、英语、seed 42，文本 11 词，输出 4.64 秒，记录的生成耗时为 **260.048 秒（约 260.05 秒，包含首次模型加载）**。输出 SHA-256 为 `e156a550dcb1aa221cd6403655ba7d54e3a035457a4f34ca42d50ec06ba52122`。ASR advisory 记录 11/11 词匹配，物理停顿检查通过。[job](../outputs/chatterbox-validation/20260929-065234-6fc5b7/deployment_attempts/offline-loader-smoke/job.json)、[WAV](../outputs/chatterbox-validation/20260929-065234-6fc5b7/deployment_attempts/offline-loader-smoke/first-success.wav)、[检查报告](../outputs/chatterbox-validation/20260929-065234-6fc5b7/deployment_attempts/offline-loader-smoke/integrity.json)

该样本属于部署排障阶段的旧能力指纹 `94fdeec79c79e62f4b3604e16fbeb12ea1c224c7b5b8b0a841169109de1e57e8`，不是上表已验收构建的产物。它只证明当时的部署可以真实生成这一短句；其冷启动耗时不能充当热推理基准，也不能推断长稿吞吐、稳定性或最终视频质量。

## 完整 52 词音频与 Pocket 对照

候选构建在独立目录 `outputs/chatterbox-validation/20260929-065234-6fc5b7/` 开始验收。固定脚本为 52 词、三个物理段落，当前切分为四个 TTS 请求；配置为 Jane、Swiss 模板、4K30、`morning-briefing` 音乐与 `morning-brief` 片尾，烧录字幕关闭以匹配选定 Daily News 基线，仍保留正常 SRT 导出。该样本不发布，不复用既有成片的镜头、审核或对话检查点。[验收请求](../outputs/chatterbox-validation/20260929-065234-6fc5b7/validation_request.json)、[运行日志](../outputs/chatterbox-validation/20260929-065234-6fc5b7/driver.log)

四段 Chatterbox 配音现已全部接受，并与同源 Pocket/Jane 配音核对。两份原稿的 SHA-256 均为 `d48408e0e96a0475f7b262f6a895867802d54e2f5d5339bda98be318c054c44c`。两方的文本/音频哈希、完整分块覆盖、manifest 契约及输出 PCM 等于原始分段精确拼接均通过，均保持自然 1.0×，没有为对齐时长而拉伸语音。[完整对照记录](../outputs/chatterbox-deployment-20260929/comparison.json)

| 指标 | Chatterbox / Jane 参考克隆 | Pocket / 内置 Jane |
| --- | --- | --- |
| 已接受原稿覆盖 | 52/52 词，4 段 | 52/52 词，3 段 |
| WAV | 24 kHz、单声道、PCM16 | 24 kHz、单声道、PCM16 |
| 完整音频时长 | 20.44 秒 | 20.28 秒 |
| 实际 ASR 精确匹配 | 52/52 词；0 段告警 | 49/52 词；1 段 advisory 告警 |
| 分段内部停顿检查 | 全部通过；最长 0.56 秒 | 全部通过；最长 0.69 秒 |
| 应用拼接点停顿 | 0.48 / 1.00 / 0.56 秒 | 0.95 / 0.60 秒 |
| 已接受分段记录耗时合计 | 833.844781 秒 | 60.963 秒 |

两份音频时长相差 0.16 秒。ASR 精确匹配与已接受原稿覆盖是不同指标：Pocket 的末段识别告警按现行 advisory 策略保留，不自动证明实际漏读；这些数据也不构成音色或主观听感排序。内部停顿统计从 0.5 秒起记录，拼接点停顿单独报告。[Chatterbox manifest](../outputs/chatterbox-validation/20260929-065234-6fc5b7/audio/tts_manifest.json)、[Pocket manifest](../outputs/chatterbox-deployment-20260929/pocket-comparison/audio/tts_manifest.json)

耗时边界不同：Chatterbox 记录服务端已接受 job 的开始到结束，首段 256.777009 秒包含新 worker 冷加载，合计不含排队、下载、ASR 或失败 job；Pocket 记录客户端 HTTP 合成耗时，包含请求重试和响应规范化、不含 ASR，模型是否热加载未单独记录。因此上述合计不能用于受控热推理速度比较。

完整音频 SHA-256：Chatterbox 为 `e74a06daec6be1036ca097b1e23b39186922ff1c28d6f741286b043dc7435505`，Pocket 为 `b3e3b8f1f50169d93059522db393cc100a2ae64170d3641e20ea1bfd841702c0`。[Chatterbox WAV](../outputs/chatterbox-validation/20260929-065234-6fc5b7/audio/tts_input_generated.wav)、[Pocket WAV](../outputs/chatterbox-deployment-20260929/pocket-comparison/audio/tts_input_generated.wav)

## 数字与品牌名补测

另用完整 40 词脚本实测 `OpenAI`、`NVIDIA`、`Google`、`ARM`、`23`、`24,000`、`30` 和最后一句结束边界。两段真实请求分别输出 11.56 秒、4.72 秒，合计 16.28 秒；source coverage 40/40、ASR 匹配 40/40、无 advisory 警告，物理音频检查通过。服务记录生成时间为 368.0345 和 177.4145 秒，客户端总流程 573.2788 秒（包括下载和校验）。这些是同次已加载模型的实际样本，不能推广为所有专名或数字读法的保证。[脚本](../outputs/chatterbox-deployment-20260929/pronunciation-check.txt)、[结果](../outputs/chatterbox-deployment-20260929/pronunciation-check-result.json)、[音频 manifest](../outputs/chatterbox-deployment-20260929/pronunciation-check-audio/tts_manifest.json)

完整补测 WAV 为 24 kHz、单声道 PCM16，SHA-256 `70091e8b6a01ce2bf6caa6fd11c41019beb2aa68abaf37faefc1d71c1c1f646a`。

## 同文受控回归：20 词长尾样本已修复

新 job `b66f1ff459034c8999f50a9d8b9aa451` 与旧失败 job 使用完全相同的 20 词原文、Jane 参考、seed 42、语言及生成参数；固定源码/权重版本不变，部署变更为恢复 Space 对齐/EOS、eager 注意力及兼容处理。[新请求与音频元数据](../outputs/chatterbox-validation/20260929-065234-6fc5b7/audio/tts_input_part_002_generated.json)

| 记录 | 旧候选 `1a4cfe95…` | 新候选 `c21819f5…` |
| --- | --- | --- |
| 服务记录的生成耗时 | 868.3809 秒 | 267.0208 秒 |
| 输出音频时长 | 32.44 秒 | 7.96 秒 |
| WAV 格式 | 24 kHz、单声道、PCM16 | 24 kHz、单声道、PCM16 |
| 内部停顿检查 | 2.78 秒，超过 0.80 秒阈值，拒绝 | 0 段、`max_seconds=0.0`，通过；检测报告阈值为 0.5 秒 |
| ASR 证据 | 额外内容和回读线索 | advisory 报告 20/20 词匹配、首尾锚点完整、无警告 |

新 WAV SHA-256 为 `fccf24dfcda5e19b37f77ae9d9f64e42dbced0c8df794c9c1124733a246266da`。[新 WAV](../outputs/chatterbox-validation/20260929-065234-6fc5b7/audio/tts_input_part_002_generated.wav)、[新检查报告](../outputs/chatterbox-validation/20260929-065234-6fc5b7/audio/verification/tts_input_part_002/integrity_report.json)、[新 ASR 时间戳](../outputs/chatterbox-validation/20260929-065234-6fc5b7/audio/verification/tts_input_part_002/transcript.json)

该受控回归证明本部署的兼容修复消除了此样本的长尾，并通过既定检查。它不单独分离 EOS 机制和 eager 的各自贡献，也不证明所有文本或语言均稳定、主观听感等同旧声音；两次耗时不作全模型吞吐结论。

## 失败历史：未恢复对齐机制的旧候选

旧候选指纹 `1a4cfe95c214dc6cfaf5941f830e57f5580c9cdcacfec0d98636849821b36d83` 对应 image `sha256:3e24b55734e635ce94a845785155e740979c93c69a420dccfc78775582fc88d6`、service SHA-256 `429daf76c0a39dd695261fe54c8e092dd66d2a909349e1ea4621b5e5769fa4be`。其 20 词 job `7cace8060e01428e8c4551a6e794d407` 曾采样超过 700 个语音 token，最终服务状态为 `completed`，产生 **32.44 秒**音频，生成耗时 **868.3809 秒**。这是服务生成完成，随后项目音频检查明确拒绝该输出，未将其用于验收成片。[旧能力记录](../outputs/chatterbox-validation/20260929-065234-6fc5b7/deployment_attempts/v2-without-alignment/capabilities.json)、[job](../outputs/chatterbox-validation/20260929-065234-6fc5b7/deployment_attempts/v2-without-alignment/long-job.json)、[原始 WAV](../outputs/chatterbox-validation/20260929-065234-6fc5b7/deployment_attempts/v2-without-alignment/long-output.wav)

物理连续性检查测得内部停顿 **2.78 秒**，超过 **0.80 秒**阈值。ASR 同时给出明显超出 20 词原文的额外内容及回读线索；这是异常证据，不将其中附加词的拼写或语言识别当作逐词准确的听感确认。[拒绝摘要](../outputs/chatterbox-validation/20260929-065234-6fc5b7/deployment_attempts/v2-without-alignment/long-integrity-summary.json)、[ASR 时间戳记录](../outputs/chatterbox-validation/20260929-065234-6fc5b7/deployment_attempts/v2-without-alignment/long-verification/transcript.json)

独立源码对照发现 GitHub `5de7a54…` 删除了指定 Space `c612a94…` 的 `AlignmentStreamAnalyzer`；该分析器根据文本/语音注意力抑制过早 EOS，并对长尾、回读及重复 token 强制 EOS，属于生成算法。新候选已恢复此机制并适配 Transformers 5.2 eager 注意力和 hooks 清理；同文回归结果见上节，机制证据见[研究文档](chatterbox-research.md#新发现space-的对齐驱动-eos-与当前-github-不同)。

## 最终视频验收

正式成片为 [video-7dd117a0a250c2e2.mp4](../outputs/chatterbox-validation/20260929-065234-6fc5b7/video-7dd117a0a250c2e2.mp4)，45,018,817 bytes，SHA-256 `7dd117a0a250c2e2e99368db48d11fc3f5b0d9aeb65b084bf3b20c5d9fdfa2ba`。FFprobe 为 H.264、3840×2160、30 fps，AAC 48 kHz 双声道，容器时长 26.986667 秒。正常导出 [SRT](../outputs/chatterbox-validation/20260929-065234-6fc5b7/video-7dd117a0a250c2e2.srt)，烧录字幕关闭，与所选基线一致。本次为隔离验收样片，没有创建应用发布任务或发布到外部平台。

| 检查 | 实际结果 |
| --- | --- |
| 正式生产质量状态 | `passed=true`、`quality_status=passed`、`delivery_status=completed`，成片哈希匹配 |
| 稿件/时间轴 | 52/52 词覆盖；3 个物理段落按实测音频安排 |
| 原旁白 → 节奏插入后旁白 | 三段帧数完全相等，比例 1.0，零偏移；波形相关系数最低 0.999987 |
| 最终混音 → 视频 AAC | 相关系数 0.999739；固定编码偏移 42.667 ms，未做变速对齐 |
| 音乐及视频 | 原有 pacing/music mix 均通过；FFprobe 与全视频、全音轨严格解码通过 |
| 多模态审核 | 实际 provider 为 ChatGPT 后备，确认接收图片；三个场景 92/98/96，平均 95.33，无告警、无 calibration 例外 |
| HTTP 正式地址 | 完整 GET 200 且 SHA 匹配；Range GET 206、前 1024 bytes 匹配 |
| Ego Lite 正式地址连续播放 | 从 0 到 26.986667 秒，1.0 倍速、未静音、无错误、零掉帧，15 个实测状态采样 |

以上结果分别见 [完整验收报告](../outputs/chatterbox-validation/20260929-065234-6fc5b7/validation_report.json)、[生产质量报告](../outputs/chatterbox-validation/20260929-065234-6fc5b7/av_sync_report.json)、[HTTP 读回](../outputs/chatterbox-validation/20260929-065234-6fc5b7/http_readback.json)、[浏览器播放](../outputs/chatterbox-validation/20260929-065234-6fc5b7/browser-playback.json)。现有质量报告顶层 `analyzer` 保留历史名称 `gemini-web-via-opencli`，本次真实审核来源应读取 `batches[].review_provider=chatgpt`，不能写成 Gemini 审核成功。

首次 Gemini 网页审核遇到模型选择器或空上传菜单，三次均在提示提交前失败；现有后备路径随后发现 ChatGPT 新 DOM 已有附件，但旧预览检测只查 `#prompt-textarea`、`button` 和旧发送属性。已修复当前 composer 作用域、真实已加载图片、同名自动编号 `(n)`、`role=button` 和实际 Send 按钮识别；未改变审核提示、模型、评分阈值或请求间隔。正常后备重试通过，复用原候选视频，没有重新生成 TTS 或重渲染。附件检测在真实 Ego Lite 页面只读验证为 true，回归 220 passed、1 既有 skip、0 failed。[OpenCLI 测试日志](../outputs/chatterbox-deployment-20260929/opencli-preview-regressions.log)

客户端最后还修复损坏请求 checkpoint 被当成空文件的问题：仅文件不存在可以创建新幂等身份；损坏、不可读或结构无效时保留原记录并停止，防止重复执行未知状态的旧请求。对应恢复回归已包含在 403 项后端测试中。验收后检查确认 nr-test 三项 TTS 均 healthy；Pocket/Orpheus 的启动时间仍为 2026-09-09，未因本次部署重启。[远端服务状态](../outputs/chatterbox-deployment-20260929/final-remote-services.json)

本次高置信确认部署、英语接入、这些实际脚本与既有制作契约一致。未完成全语言视频认证、小时级长稿压力测试、受控冷热吞吐基准或主观音色盲评。CPU 合成耗时与参考克隆的音色差异是真实限制，因此保留现有默认配音选择。
