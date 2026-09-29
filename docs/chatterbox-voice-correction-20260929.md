# Chatterbox 官方英语声音修复：2026-09-29

## 原因与修复范围

上一版把“保持现有视频效果”错误解释为保留 Pocket/Jane 音色，将已有 Pocket/Jane 的旁白直接作为 Chatterbox 参考。这确实会克隆出接近 Pocket/Jane 的声音，而不是用户指定的官方 Space 英语声音。服务端任务、返回音频 SHA 与应用缓存读回确认实际调用了 Chatterbox；问题在参考选择，并非请求串到了 Pocket。

历史 Jane 参考为 `outputs/20260928-235341-4aa80a/audio/tts_input_part_001_generated.wav`，SHA-256 `4c3f950e7e549b41143921da5e5d8e477573d4104d74b03f70fcdeb014a10060`。旧部署 `c21819f5…` 的视频制作验收有效，但不能被解释为官方声音验收。该样片及其所有检查点保留。[历史部署记录](chatterbox-deployment-20260929.md)

新默认声音为 `space-en-f1`，界面显示 **Official Space English**。它使用固定 [Space app.py](https://huggingface.co/spaces/ResembleAI/Chatterbox-Multilingual-TTS/blob/c612a94210c0af895d6b649850d15b7ccca36594/app.py) 英语分支的原始 `en_f1.flac`：

| 参考身份 | 已核对的值 |
| --- | --- |
| GCS generation | `1755807478904629` |
| 大小 | 83,323 bytes |
| SHA-256 | `e81950378b53827aed08d164ebd332a7b8c5805b8a97b4f79ab057071e359c72` |
| 格式 | FLAC、24 kHz、单声道、16-bit、72,936 samples / 3.039 秒 |
| 原件 | [固定对象版本](https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac?generation=1755807478904629) |

官方 Space 首页初始语言为 French；本项目交付链当前使用 English，本次核对官方英语分支。选择不同语言或上传自己的参考会改变声音，不能以另一个参考的试听判断英语默认声线。来源、源码与条件编码流程的逐项核对见[研究记录](chatterbox-research.md)。

显式 `jane` 保留历史语义并显示 **Jane (Pocket clone)**；显式 `default` 保留 `conds.pt` 并显示 **Model default voice**。没有将历史 ID 重新指向不同声音。切换到 Chatterbox 时使用模型默认 `space-en-f1`，避免沿用上一个模型同名 Jane。试听 URL 绑定版本，旧浏览器缓存不再覆盖新部署声音。

## nr-test 实际构建

2026-09-29 09:41:50 UTC 新容器启动，仍为独立 `10.60.11.3:8092` 服务。官方参考启动时及每个任务前都校验固定大小和 SHA；能力响应同时公开默认声音、参考来源版本、声音指纹和部署指纹。旧 `jane`、`default`、任务记录、音频和镜像均保留。

| 构建项 | 实际值 |
| --- | --- |
| Docker tag | `chatterbox-tts:5de7a54-v2-space-en` |
| Image ID | `sha256:da29c68bfaf4e86b565bc08a315364a0411158e832e82d7af951b4d68ed6027d` |
| 服务 SHA-256 | `bbc62ef1a4b257bf0c76f8c74fff3b90948adc31bc3655339a7c95187a451137` |
| 部署指纹 | `d5d1adcf03462accae576565ab8096a283de47c3d4d0b1caf52ade0bf9d04218` |
| 官方声音指纹 | `f65b821cb8b0fe7af455aa02c4078a644b4f1d53801a01fc452e1f03f79056bf` |

模型、权重、Space EOS/对齐算法、eager 兼容补丁和六项合成参数与前一已验收构建相同。本次不借机修改语速或解码行为。Pocket 与 Orpheus 启动时间仍为 2026-09-09；原 Morning Desk 配置保持 Pocket/Jane、自动化关闭、`auto_publish=false`。

能力记录：[capabilities.json](../outputs/chatterbox-voice-correction-20260929/capabilities.json)。官方原件与独立来源核对保存在 `outputs/chatterbox-deployment-20260929/official-space-reference-audit/`。运行证据与媒体均为 ignored 工作区产物，不进入 Git。

## 实际合成与试听

首段实际 job `f199e50a79bb4ea88db37b1b6c1447c5` 使用 `space-en-f1`、English、seed 42，输出 4.48 秒；ASR 精确匹配 11/11 词，物理停顿检查通过。使用同一个已完成请求的幂等键、仅省略 `voice` 重新提交，服务仍返回同一个 job，并明确将声音解析为 `space-en-f1`，没有新生成任务。这验证了实际 HTTP 默认行为，而不只是目录排序。[首段任务](../outputs/chatterbox-voice-correction-20260929/first-job.json)、[默认行为读回](../outputs/chatterbox-voice-correction-20260929/default-voice-http-proof.json)

正常试听生成器产生的新音频为 2.68 秒，SHA-256 `47b47dc2df70c885d1aedab5b81cba45386a0b4c6493444dc55270b10b8a9dcb`。远端 job `37d7f7a006d8436bb4d295be2a756f45` 返回的字节与本地缓存完全相同；旧 Chatterbox/Jane 试听仍为 3.12 秒、SHA-256 `26ae574b4d3ea3b2cede5e7f6deada0756c87089e4912ef473194ba111440450`。这些哈希用于确认路由和音频来源，不代替主观音色判断。[新试听 WAV](../outputs/chatterbox-voice-correction-20260929/official-space-en-preview.wav)、[试听来源对照](../outputs/chatterbox-voice-correction-20260929/preview-source-comparison.json)

独立复核额外发现试听发布的竞态：生成过程中修改 Admin 参数，路由虽然最终拒绝响应，音频仍可能先进入旧缓存目录。客户端现固定本次生成参数快照并写入 manifest；试听发布前核对初始、当前、实际生成契约、声音、语言和音频 SHA，拒绝将不匹配音频写入旧目录。模型目录的版本字段进入浏览器 URL，并定期刷新，正在打开的页面也能发现变更。

## 代码与界面回归

本次定向检查合计 **602 tests passed、183 subtests passed**；`npm run build`、`npm run lint`、`git diff --check` 通过。覆盖默认选择及历史恢复、参考篡改拒绝、服务队列与幂等、自然速度与停顿、A→B/ABA 试听缓存竞态。唯一测试警告为既有 `twitter_text` 对 `pkg_resources` 的弃用依赖。[应用检查](../outputs/chatterbox-voice-correction-20260929/checks/application-tests.log)、[服务与连续性检查](../outputs/chatterbox-voice-correction-20260929/checks/service-continuity-tests.log)、[前端构建](../outputs/chatterbox-voice-correction-20260929/checks/frontend-build.log)、[lint](../outputs/chatterbox-voice-correction-20260929/checks/frontend-lint.log)

09:51 UTC 确认实际生产任务与 Explainer 队列为空后，重启本地应用加载新代码。Ego Lite 在 Morning Desk 从原 Pocket/Jane 切到 Chatterbox，实际选中 `space-en-f1`，三个选项标签与预览按钮均正确。这是未保存的界面验证，没有改动 Morning Desk 已保存配方。[界面切换证据](../outputs/chatterbox-voice-correction-20260929/browser-model-switch.json)

## 末句 ASR 告警的独立复核

本次 52 词样片四段均通过物理音频检查，末段 8 词的整段 ASR 只返回后 5 词，现有 advisory 告警原样保留。独立诊断对原 WAV 的前 0.76 秒不变速截取，不提供原稿提示，识别出 `Thanks for watching.`；加静音的同段和前 1.1 秒截取也得到该句。后 0.76–2.28 秒单独识别出后五词，开头波形确有语音能量。重复开头的诊断副本识别出该句两遍。

这些证据支持整段 ASR 漏识别首句，不支持直接判定 TTS 漏读。空白对照会幻觉 `Thank you.`，因此也不把单个短片段 ASR 当成无条件证据。没有修改原 WAV、源文、既有告警或质量阈值，没有为消除告警重生成已接受音频。[独立诊断](../outputs/chatterbox-voice-correction-20260929/diagnostics/asr_part004.json)、[裁片及空白对照](../outputs/chatterbox-voice-correction-20260929/diagnostics/asr_part004_controls.json)

## 最终视频及应用启用

本次独立样片为 [video-1193ec9a0634fd6f.mp4](../outputs/chatterbox-validation/20260929-094205-faef83/video-1193ec9a0634fd6f.mp4)，41,301,927 bytes，SHA-256 `1193ec9a0634fd6f352c378b3447e28b77a9bdd32215cdd10f02988294989c01`。使用官方 `space-en-f1`，旁白 18.44 秒，成片 24.987696 秒，H.264 / 3840×2160 / 30 fps、AAC 48 kHz 双声道。与已有生产设置一致，烧录字幕关闭，保留正常 SRT、音乐、片头片尾与实测音频时间轴。该样片没有创建发布任务，也没有发布到外部平台。

| 检查 | 实际结果 |
| --- | --- |
| 正式状态 | `passed=true`、`quality_status=passed`、`delivery_status=completed`；正式文件与报告哈希匹配 |
| 合成来源 | 四段均绑定官方参考指纹；52 词完整请求覆盖，整段 ASR 实际识别 49/52，末句复核见上节 |
| 原旁白 → paced narration | 各段帧数比 1.0、偏移 0；相关系数最低 0.99999103，无时间拉伸 |
| 混音 → 成片 AAC | 相关系数 0.99980728，固定编码偏移 42.667 ms |
| 制作检查 | 停顿、节目节奏、混音、FFprobe、全视频及音频严格解码通过 |
| 多模态审核 | 实际 Gemini、图片接收已确认；3/3 场景通过，平均 82/100，无 calibration 例外 |
| HTTP | 完整 GET 200、SHA 匹配；Range GET 206、首 1024 bytes 匹配 |
| Ego Lite 连续播放 | 14 次状态采样，从 0 至 24.987696 秒；1.0 倍速、未静音、无播放错误、零掉帧 |

以上分别见[完整验收](../outputs/chatterbox-validation/20260929-094205-faef83/validation_report.json)、[生产质量报告](../outputs/chatterbox-validation/20260929-094205-faef83/av_sync_report.json)、[HTTP 读回](../outputs/chatterbox-validation/20260929-094205-faef83/http_readback.json)、[连续播放](../outputs/chatterbox-validation/20260929-094205-faef83/browser-playback.json)、[播放质量](../outputs/chatterbox-validation/20260929-094205-faef83/browser-playback-quality.json)。自动生成的验收报告保留“浏览器待另行检查”字段，独立浏览器记录已补齐该检查，没有篡改原报告。

完整验收后已通过应用设置接口启用 `d5d1adcf…`，并重新加载应用以显示最终缩短后的声音标签。版本化试听 GET 200，SHA 与远端官方参考任务返回字节一致；旧版本 URL 返回 409。Ego Lite 点击 **Preview Official Space English**，实际从 0 播放至 2.68 秒结束，1.0 倍速、未静音、无错误。[启用及设置读回](../outputs/chatterbox-voice-correction-20260929/application-activation.json)、[页面按钮试听](../outputs/chatterbox-voice-correction-20260929/browser-preview-playback.json)

Morning Desk 已保存设置逐字段与修复前相同：Pocket/Jane、自动化关闭、`auto_publish=false`；全局默认模型仍为 VibeVoice 0.5B。nr-test 三项 TTS 均 healthy，Pocket/Orpheus 的镜像和启动时间未变。[远端最终状态](../outputs/chatterbox-voice-correction-20260929/final-remote-services.txt)

高置信确认本项目英语分支现在使用与指定 Space 相同的官方参考原件、V2 模型和默认生成参数，并完成真实视频验收。CPU 与 Space GPU、依赖及随机采样存在差异，未把来源一致表述为逐样本波形或主观听感完全相同；本次也没有将代表性样片扩大为全语言或长稿认证。
