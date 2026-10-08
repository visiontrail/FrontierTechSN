# 长篇图解工作台

独立入口：侧边栏 **长篇图解**，路径 `/explainer-studio`。这条工作流从用户提供的调研材料开始，不接入新闻采集、公共素材搜索、新闻审核或自动发布。

## 使用

1. 新建项目，填写标题并粘贴调研，或导入 UTF-8 `.txt` / `.md`。资料不自动联网补充；脚本模型被要求依据已给材料保留事实强度、不编造研究。
2. 选择语言、目标时长、配音模型及声音。新项目默认使用 **Kyutai Pocket TTS**，音色为 **alba**，独立于新闻工作流的默认模型；另存的工作台默认设置优先，已有项目和本地草稿保留自己的选择。Orpheus English、Pocket English 和 VibeVoice Realtime 0.5B 按英语模型处理；中文可选现有 VibeVoice 1.5B。新功能尚未完成中文实际配音验收。
3. 修改画风、人物说明、色板和写作语气。可另存为以后新项目的默认设置，已有项目保留自己的快照。
4. 在 **Research & direction** 保存选题，通过 **Save & add another** 连续规划多个选题。每个选题保留自己的调研材料、语言、声音和画风；保存本身不会触发生成。
5. 点击 **Add to video queue**，将选题加入生成队列。如果没有分镜，系统先根据调研规划章节并生成分镜，再自动完成配音、ChatGPT 插图、可选 Gemini 动画和视频合成。有分镜的项目直接使用已保存的分镜。目标时长用于写作预算；最终时长以真实配音为准。
6. 如需先编辑脚本，可单独加入脚本生成队列；完成后直接编辑、增加、删除分镜，再加入视频生成队列。完成后在同页播放或下载视频、音频和时间轴。

### 选题计划与生成队列

工作台顶部的 **Topic plans** 展示可继续编辑的选题，**Video queue** 展示等待项目的顺序、当前执行任务和等待原因。选题按实际入队顺序依次执行，不设置生成时间。一个视频任务的脚本与视频阶段连续执行，期间不会领取其他视频任务。

Morning Desk 和 Explainer Studio 共用一个后台消费者。正在执行的任务不会被抢占；它结束后先检查到期的 Morning Desk 任务，再领取下一项 Explainer。未来定时的新闻不会阻塞当前可执行的选题。此规则涵盖单独生成脚本、单独生成视频以及完整生成三个入口。

排队中的选题不可编辑、不可重复入队。**Remove from queue** 将其恢复到入队前的状态，已有成片和材料保留；修改后再次入队会排到队尾。如果任务已被领取，移出请求返回 409 并刷新状态，不会意外取消正在执行的任务。执行中的项目可通过 **Cancel run** 取消，等待当前任务清理结束后继续下一项。一个项目失败不会卡住后续队列；重试需要重新入队，可复用已验证的缓存。

队列与操作类型保存在项目 JSON 中。服务重启保留尚未领取的排队项目；已经执行的项目标记为中断或取消，不自动重新提交可能已经被外部提供方接受的请求。运行方式仍限定为单个 ASGI 进程。

默认暖纸色、简笔人物、静态硬切、每镜约四秒、无烧录字幕。关键词由渲染器添加，避免依赖图像模型拼写。可改为轻推拉或 Gemini 动态；Gemini 默认仅处理显式选中的镜头，也可设置每 N 镜头自动生成动态（0 表示仅手动）。字幕可开启，同时总会导出按镜头配音时间生成的 SRT。

镜头目标时长是写作引导，不是固定裁切时长。每个镜头使用自己的实际配音长度，最多补足一个视频帧的静音。不同图像无法保证像素级角色身份一致；重复的人物设定控制整体一致性。

## 现有能力盘点与复用

本表来自本次代码检查，不表示每个外部服务均在本次完成实测。

| 能力 | 现有入口 | 在长篇图解中的使用 |
| --- | --- | --- |
| 写作、结构化输出、多模型路由、重试与凭据轮转 | `pipeline/digester.py`、`agent.py`、`model_router.py`、`providers.py` | 复用配置和请求层，新增章节/分镜写作任务 |
| ChatGPT 网页文生图与下载 | `collage_broll.py:_generate_still`、`thumbnail.py`、OpenCLI ChatGPT adapter | 按分镜生成原创手绘图；使用独立画风提示词 |
| Gemini 参考图片生成视频 | `collage_broll.py:_generate_video`、`tools/opencli/patches/gemini-video-command.js` | 可选镜头动效；支持单参考帧，保留既有双关键帧调用、任务内下载和会话来源 |
| 请求节流、访问限流、浏览器会话隔离与回收 | `opencli.py`、`opencli_rate_limit.py`、`opencli_session.py` | 原样复用，不绕过配置或抢占其他任务会话 |
| Orpheus、Pocket TTS、VibeVoice 配音 | `pipeline/tts.py`、`config.TTS_MODELS`、`routers/voices.py` | 单镜配音、原生语速、声音选项与既有音频完整性机制 |
| 音频测量、FFmpeg 合成、媒体验证 | 既有媒体模块和系统 FFmpeg | 独立的确定性图片合成器，逐镜帧对齐、全片解码 |
| HyperFrames HTML 场景、字幕、转场与渲染 | `hyperframe/`、`composer.py`、`director.py`、`scene_kit.py` | 本版简单切图用 FFmpeg，保留现有 HyperFrames 供其他节目使用 |
| 新闻研究、事实核对、素材获取与适配 | `daily_news/`、`news_images.py`、`web_footage.py`、`review_response.py` | 不接入新流程 |
| 缩略图、标题策略、社交文案 | `pipeline/thumbnail.py`、`title_strategy.py`、`social_copy.py` | 能力存在，当前新功能范围截至长视频成片 |
| 音乐库和 Gemini 音乐、片头片尾、品牌包装 | `music.py`、`intros.py`、`outros.py` 及相关模块 | 保留既有能力；新功能不自动加新闻包装 |
| YouTube/X/播客发布、发布记录、排程 | `publishing.py`、`account_ops/`、`daily_news/scheduler.py` | 不从图解项目触发发布；避免扩大本次范围 |

## 产物和恢复

项目保存在 `outputs/explainers/exp-<id>/project.json`，全局默认值在 `outputs/explainers/defaults.json`。这些运行产物不进入 Git。任务含保存版本号：排队和运行中拒绝编辑，过期页面保存返回 409，以免覆盖新数据。

逐章脚本在 `writing/<input-hash>/` 保存。音频、图片、动态、镜头片段分别按内容哈希保存在 `artifacts/<scene>/`；有效产物包含文件校验值与完成记录。修改旁白、声音、画风或镜头内容会使相关缓存失效。取消或服务重启保留已有文件；点击相应生成按钮继续，完成部分会复用。

Gemini 新请求保存独立提交标识和完整提示词。当 Videos 页面不更新时，可以从新增历史对话中寻找唯一且完整匹配的单轮请求，重新打开原结果。历史记录未加载、匹配不唯一、旧任务缺少标识或对话包含多轮请求时，不会据此认领视频。已提交但结果未知的请求不会盲目重新提交；已核验链接的恢复保留原有截止时间。

最终输出是 1920×1080 / 30 fps MP4、拼接 WAV、`timeline.json`、`captions.srt` 和 `manifest.json`。时间轴记录每镜对应原文、帧边界、实测语音时长、资源路径、哈希、图片和动态会话 URL。Gemini 片段只播一次，时长不足时保持末帧；长于旁白时将完整无声动作加速到镜头长度，避免截掉动作结尾。旁白始终保持原生速度。

本机 FFmpeg 没有 libass，因此字幕和关键词用 Pillow 生成透明文字层再合成。字幕按整个镜头显示，不是逐字卡拉 OK 字幕。开启字幕时过长的一段会要求拆镜头，以免文字溢出。

### English sample project copy

The English interface displays saved project titles and research verbatim. Earlier validation samples retained Chinese titles after the interface was translated. To translate those specific samples, preview the changes with `.venv/bin/python scripts/translate_explainer_samples.py`, then apply them with `--apply`. The script targets the known sample IDs and original text only; it does not translate arbitrary user content or change the selected narration language.

Original project JSON is backed up under `outputs/explainer-validation/` before replacement. The cleanup preserves completed status, generated assets, progress, scene content and timestamps, and increments the revision to reject stale saves. It also translates the old sample title embedded in research and the DeepSeek display name in the sample run log. Active projects are rejected. The running service reads these files on each request, so the library updates without rebuilding or restarting the service.

## 耗时与运行约束

每张插图在进入合成前必须通过视觉逻辑审核。分镜、生成提示词和审核共享 `backend/prompts/explainer_visual_logic.txt`，可在 Admin → Prompts 的 **Explainer visual logic** 中调整。规则覆盖透视与遮挡、物体结构、身体与物体的交互、物理因果和旁白含义，不按特定道具列补丁。简笔画和明确的视觉隐喻允许简化，但不能掩盖普通物体的结构或动作矛盾。

审核把实际图片交给 Gemini 网页视觉模型，逐项要求可见证据；审核不可用时才尝试现有 ChatGPT 视觉备用链路。任一项失败或不确定都不能进入合成。结构完整的否决不会换一个模型重新寻求通过。失败反馈会加入下一张候选的提示词，每个场景最多生成 3 张候选；仍不合格或两个审核提供方都不可用时任务停止，保留原图、响应、报告和配音供续跑。审核增加了真实模型调用和等待时间。

审核报告绑定图片 SHA-256、场景内容和规则版本。旧版缓存图片需先补审，通过后可复用；被否决的图片重做。原提供方文件不会覆盖。审核中断后续跑复用已生成候选，不会因缺少审核结果就重新生成图片。模型审核降低漏检风险，但不等于能保证检测到所有视觉错误。

ChatGPT 生成沿用现有 10—30 分钟请求间隔（当前配置 600 秒），Gemini 沿用 60—120 秒随机间隔。180 张全新插图仅请求间隔下限就约 29.8 小时，尚未计入图像生成、审核和配音。工作台会显示实际镜头数对应的下限。较长的每镜时长可减少图量，但会改变参考视频的快切节奏；缓存可减少重试成本。

本版是一镜一图的生成路径，没有实现联系表批量裁图或图库检索。批量图片复用与多镜共用连贯配音是后续提高吞吐量、保持口播连贯性的明确优化空间。

与现有应用相同，仅运行一个 ASGI 进程；多 worker 共享同一产物目录不受支持。应用退出会取消本进程图解作业；启动时将中断状态标记为可重试，绝不将未完成作业显示为成片。

## 验证

```sh
.venv/bin/python -m pytest backend/tests/test_explainers.py backend/tests/test_explainer_pipeline.py backend/tests/test_studio_worker_queue.py -q
cd frontend
npm run lint
npm run build
```

测试覆盖 API 保存与冲突、模型语言匹配、生成错误、取消与重启、章节结构修复、脚本缓存、真实 FFmpeg 1080p 合成/完整解码、音轨采样与帧边界、静态/推拉/Gemini 短片保持、资源缓存失效及路径约束。外部模型和浏览器调用在这些测试中使用替身；实际服务验收另记于运行产物，不能将单元测试等同于整条真实服务链路通过。

### 2026-09-24 实际服务验收

本机 `http://localhost:8101` 已启用工作台，并完成以下真实调用：

- 写作：以明确标注的虚构示例材料调用已配置的模型，生成 15 镜头、144 个英文词的可编辑脚本；目标时长为 1 分钟。
- 静态样片：`exp-f10de11e918f494d80c3f6c0`，真实 Orpheus 配音、OpenCLI ChatGPT 插图及 FFmpeg 合成，7.0 秒。它复用了下面动态样片已生成并校验的同一份音频、原图，未重复调用提供方。
- 动态样片：`exp-263d1cd695764133a89bdf7a`，同一 ChatGPT 原图经 Gemini 单参考图生成 10 秒动作，完整适配至 7.0 秒镜头；旁白实长 6.997333 秒，未变速。
- 两支成片均通过完整解码、HTTP 下载、Range 206 与文件哈希比对，并在 Ego Lite 工作台播放器实际播放至结束，播放器无错误。
- 桌面明暗主题及 390 px 手机布局已检查；材料导入、旁白导出、保存后重新进入、默认风格保存和未保存编辑的轮询保护均已检查。

动态请求曾在 Gemini `/videos` 页停留于“正在生成”，本地 600 秒无进度检查退出。重新打开与完整提示词匹配的历史对话后，确认原请求已经产出视频；补回已核验的对话链接后续传成功，没有提交替代生成。它证明了素材生成和恢复后的完整链路；这次人工发现历史链接不能算作修复后自动历史发现机制的实际验收，后者另由回归测试验证。

验收记录、播放状态与截图位于忽略目录 `outputs/explainer-validation/`。本次实测范围是英文写作和短样片，没有生成完整 15 分钟节目，也没有完成中文声音或多镜头角色一致性的实际验收。
