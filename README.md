# Audio_test

`麦克风采集 -> 音频增强 -> whisper.cpp 推理 -> 文本清洗 -> 事件化输出`

当前项目适合做三类事情：

- 快速验证麦克风到文字的整条链路
- 对比 `base / small` Whisper 模型效果
- 导出统一结构的转写事件，方便后续接接口、数据库或消息队列

## 功能

- 支持 `direct` 模式：先录音，再统一转写
- 支持 `stream` 模式：持续输出 `partial / final` 片段
- 支持模型发现与切换：`tiny / base / small / auto`
- 支持中文文本规范化与伪字幕过滤
- 支持音频增强：轻量增益、静音检测
- 支持运行指标统计：RMS、峰值、推理次数、segment 数量、耗时
- 支持导出：
  - `raw_audio.wav`
  - `enhanced_audio.wav`
  - `transcription.json`
  - `events.json`
- 支持统一事件模型 `TranscriptionEvent(BaseModel)`
- 支持基础单元测试

## 目录结构

- [`audio.py`](/Users/xingyujie/Desktop/code/Audio_test/audio.py)
  麦克风采集，基于 `sounddevice`
- [`audio_processing.py`](/Users/xingyujie/Desktop/code/Audio_test/audio_processing.py)
  音频能量估计、静音判断、轻量增益与裁剪
- [`config.py`](/Users/xingyujie/Desktop/code/Audio_test/config.py)
  模型与动态库发现、运行配置、模型列表解析
- [`models.py`](/Users/xingyujie/Desktop/code/Audio_test/models.py)
  数据模型，包括：
  - `TranscriptionSegment`
  - `TranscriptionMetrics`
  - `TranscriptionEvent`
- [`whisper_binding.py`](/Users/xingyujie/Desktop/code/Audio_test/whisper_binding.py)
  `whisper.cpp` 的 `ctypes` 绑定
- [`text_postprocess.py`](/Users/xingyujie/Desktop/code/Audio_test/text_postprocess.py)
  文本清洗、简体规范化、伪字幕过滤、增量文本处理
- [`transcription_engine.py`](/Users/xingyujie/Desktop/code/Audio_test/transcription_engine.py)
  流式转写引擎，输出 `partial / final`
- [`result_export.py`](/Users/xingyujie/Desktop/code/Audio_test/result_export.py)
  导出 JSON / WAV 结果
- [`run_audio_test.py`](/Users/xingyujie/Desktop/code/Audio_test/run_audio_test.py)
  CLI 主入口，负责录音、转写、指标统计、事件导出
- [`start_audio_test.sh`](/Users/xingyujie/Desktop/code/Audio_test/start_audio_test.sh)
  更短的启动脚本
- [`requirements.txt`](/Users/xingyujie/Desktop/code/Audio_test/requirements.txt)
  运行依赖
- [`tests/test_config.py`](/Users/xingyujie/Desktop/code/Audio_test/tests/test_config.py)
  模型/动态库发现测试
- [`tests/test_text_postprocess.py`](/Users/xingyujie/Desktop/code/Audio_test/tests/test_text_postprocess.py)
  文本清洗测试
- [`tests/test_transcription_engine.py`](/Users/xingyujie/Desktop/code/Audio_test/tests/test_transcription_engine.py)
  流式转写最小集成测试

## 环境准备

安装当前依赖：

```bash
cd /Users/xingyujie/Desktop/code/Audio_test
pip install -r requirements.txt
```

依赖包括：

- `numpy`
- `sounddevice`
- `opencc-python-reimplemented`
- `pydantic`

## 模型与动态库

程序会自动搜索这些位置：

- 当前项目的 `models/`
- 当前项目的 `vendor/whisper/`
- 上级目录的 `vendor/whisper/`
- 上级目录的 `SynPhonie/vendor/whisper/`
- 环境变量：
  - `SYNPHONIE_WHISPER_SEARCH_DIR`
  - `WHISPER_CPP_DIR`

也可以显式指定：

- `--model`
- `--model-path`
- `--library-path`
- `--backend`
- `SYNPHONIE_WHISPER_MODEL_PATH`
- `SYNPHONIE_WHISPER_LIBRARY_PATH`
- `SYNPHONIE_WHISPER_BACKEND`

当前本地常见模型：

- `base`: `/Users/xingyujie/Desktop/code/SynPhonie/vendor/whisper/ggml-base.bin`
- `small`: `/Users/xingyujie/Desktop/code/Audio_test/models/ggml-small.bin`

## 常用命令

列出当前发现的模型：

```bash
zsh start_audio_test.sh list
```

中文 `direct`：

```bash
zsh start_audio_test.sh direct 8 small zh
```

英文 `direct`：

```bash
zsh start_audio_test.sh direct 8 small en
```

中文 `stream`：

```bash
zsh start_audio_test.sh stream 10 small zh
```

跳过麦克风，验证模型链路：

```bash
zsh start_audio_test.sh skip-mic 0 small zh
```

显式使用 Python 启动：

```bash
python3 run_audio_test.py --mode direct --seconds 8 --model small --language zh
```

导出结果到目录：

```bash
zsh start_audio_test.sh direct 8 small zh ./artifacts/run1
```

模型对比：

```bash
python3 run_audio_test.py --mode direct --seconds 8 --model small --language zh --compare-models base,small
```

## 启动脚本参数

当前脚本格式：

```bash
zsh start_audio_test.sh [mode] [seconds] [model] [language] [output_dir]
```

参数说明：

- `mode`
  - `direct`
  - `stream`
  - `skip-mic`
  - `list`
- `seconds`
  录音时长
- `model`
  - `auto`
  - `tiny`
  - `base`
  - `small`
- `language`
  例如 `zh`、`en`
- `output_dir`
  可选，导出目录

示例：

```bash
zsh start_audio_test.sh direct 8 base zh
zsh start_audio_test.sh direct 8 small en
zsh start_audio_test.sh stream 10 small zh ./artifacts/stream_run
```

## 输出说明

CLI 终端默认会输出：

- 运行模式
- 模型路径
- 录音时长
- 音频能量：
  - `rms`
  - `peak`
- 转录文本
- 运行指标：
  - `transcribe_calls`
  - `raw_segments`
  - `cleaned_segments`
  - `filtered_segments`
  - `transcribe_time`

如果启用了导出，会额外写出文件：

- `raw_audio.wav`
- `enhanced_audio.wav`
- `transcription.json`
- `events.json`

## 接口说明

项目当前统一使用下面这个事件结构：

```python
class TranscriptionEvent(BaseModel):
    session_id: str
    event_id: str
    event_type: str = "transcription"
    source_chunk_id: str
    timestamp_start: float
    timestamp_end: float
    text: str
    language: str = "zh"
    confidence: float = 0.0
    is_final: bool
    model_name: str
    speaker_id: str | None = None
    tokens_count: int
```

字段说明：

- `session_id`
  一次运行对应一个 session
- `event_id`
  当前事件唯一 ID
- `event_type`
  当前固定为 `"transcription"`
- `source_chunk_id`
  结果来源 chunk 或 segment 标识
- `timestamp_start`
  片段起始时间
- `timestamp_end`
  片段结束时间
- `text`
  转写文本
- `language`
  当前片段语言
- `confidence`
  当前默认值为 `0.0`，后续可接真实置信度
- `is_final`
  是否最终结果
- `model_name`
  例如 `base`、`small`
- `speaker_id`
  当前固定为 `None`
- `tokens_count`
  文本 token 数量的近似统计

### `events.json`

`events.json` 只包含事件输出：

```json
{
  "events": [
    {
      "session_id": "c9f0...",
      "event_id": "f1d2...",
      "event_type": "transcription",
      "source_chunk_id": "direct-chunk-00001",
      "timestamp_start": 0.0,
      "timestamp_end": 3.5,
      "text": "今天我们来讲 Whisper。",
      "language": "zh",
      "confidence": 0.0,
      "is_final": true,
      "model_name": "small",
      "speaker_id": null,
      "tokens_count": 4
    }
  ]
}
```

### `transcription.json`

`transcription.json` 在 `events` 之外，还会包含：

- `runtime`
- `metrics`
- `results`

适合调试和归档。

## 优化日志

下面是这个实验台从最小可运行版本逐步演化出来的关键优化点。

### 1. 基础链路拆分

- 从主项目中拆出独立目录
- 保留最小依赖，方便单独验证
- 支持 `direct` 和 `stream`

### 2. 模型与动态库发现增强

- 修正默认动态库路径，兼容 `SynPhonie/vendor/whisper/`
- 支持本地 `models/` 目录优先
- 支持 `base / small` 自动发现

### 3. 本地 `small` 模型接入

- 本地新增 `ggml-small.bin`
- 启动脚本支持显式模型切换
- 支持 `list` 列出当前可发现模型

### 4. 音频增强与静音判断

- 增加轻量增益
- 增加 RMS 静音判断
- 导出增强后的音频，方便调试

### 5. 文本清洗

- 增加中文文本规范化
- 过滤伪字幕内容，例如：
  - `字幕: ...`
  - `字幕制作: ...`
- 支持简体化处理

### 6. 流式转写稳定性修复

- 修复 `stream` 结束时模型被过早释放的问题
- 区分 `partial / final`
- 增加运行指标统计

### 7. 更短的启动方式

- 新增 [`start_audio_test.sh`](/Users/xingyujie/Desktop/code/Audio_test/start_audio_test.sh)
- 支持：
  - 模式选择
  - 模型选择
  - 语言选择
  - 导出目录

### 8. 节能方向的收敛

- 删除实验性的 `auto -> zh/en` 自动回退逻辑
- 改成“运行前明确指定语言”
- 避免一次转录中重复多轮语言尝试，降低功耗和延迟

### 9. 接口化输出

- 引入 `TranscriptionEvent(BaseModel)`
- `direct / stream / skip-mic` 全部统一导出事件
- 增加独立的 `events.json`

## 当前限制

- 当前更适合“先明确语言再转录”，不再默认做自动语言融合
- `confidence` 目前仍是占位值，尚未接真实置信度
- `speaker_id` 暂未实现，说话人分离未接入
- `stream` 模式已支持 `partial / final`，但还不是完整低功耗生产级方案
- 若要做教室级长时运行，推荐继续向下面方向演进：
  - 模型常驻内存
  - 更强的 VAD
  - 停顿触发 final
  - 更完整的事件消费接口

## 测试

运行全部测试：

```bash
python3 -m unittest discover -s tests -v
```

当前覆盖内容：

- 模型发现逻辑
- 文本清洗逻辑
- 流式转写最小集成链路
