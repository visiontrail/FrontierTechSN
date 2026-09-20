# YouTube 字幕文件

每次成片通过现有音画质量检查后，程序同时生成与该版 MP4 同名的 `.srt` 文件。
例如 `video-<hash>.mp4` 对应 `video-<hash>.srt`。SRT 使用纯 UTF-8 编码、连续编号及
`HH:MM:SS,mmm --> HH:MM:SS,mmm` 时间码，符合
[YouTube 的 SubRip 格式要求](https://support.google.com/youtube/answer/2734698?hl=en)。

任务页提供 **Download Subtitles (.srt)**。上传时进入 YouTube Studio → 字幕，
选择与口播一致的语言，上传文件并选择“带时间信息”。这一步由用户操作，生成和
下载文件本身不会改动已上传的视频。

原来的 Captions 开关控制画面内字幕；独立 SRT 在两种开关状态下都会生成。
时间来自最终 storyboard，并复用画面字幕的分组和时间计算，保留片头、音乐间隔
以及正文的偏移。长句内部的小组时间按文本长度分配，不能称为逐词精确对齐。
片尾没有口播的来源滚动不生成字幕。SRT 不携带画面字幕的字体、颜色或位置。

字幕绑定具体成片版本。失败或重试期间保留的视频只能下载其配套旧版字幕，
按钮会注明 Last Validated；不能把本次草稿字幕当作旧成片字幕。缺少配套文件时
不显示下载按钮，接口返回 404，不会临时从可能已改变的稿件重新估算。

旧任务不会自动重渲染。要为旧成片补导字幕，必须先确认保存的 storyboard 确实
属于该版视频，再调用 `backend.pipeline.subtitles.write_subtitles(board, video_path)`。
不可直接给所有旧任务批量套用当前 storyboard，尤其是经历过失败重试的任务。
