可以，按下面配就行。

1. 配置转录（Deepgram）
```bash
python3 -m lite_synphonia providers add deepgram \
  --base-url https://api.deepgram.com \
  --model whisper-large \
  --api-key "$DEEPGRAM_API_KEY" \
  --service transcription
```

2. 配置摘要（比如 MiniMax）
```bash
python3 -m lite_synphonia providers add minimax \
  --base-url https://api.minimaxi.chat/v1 \
  --model MiniMax-Text-01 \
  --api-key "$MINIMAX_API_KEY" \
  --service summarization
```

3. 配置 embedding（用于 PDF 匹配）
- 如果还用 MiniMax（`embo-01`）：
```bash
python3 -m lite_synphonia providers add minimax-embed \
  --base-url https://api.minimaxi.chat/v1 \
  --model embo-01 \
  --api-key "$MINIMAX_API_KEY" \
  --service embedding
```
- 如果用 OpenAI embedding：
```bash
python3 -m lite_synphonia providers add openai-embed \
  --base-url https://api.openai.com/v1 \
  --model text-embedding-3-small \
  --api-key "$OPENAI_API_KEY" \
  --service embedding
```

4. 查看是否配置成功
```bash
python3 -m lite_synphonia providers list
python3 -m lite_synphonia providers show deepgram
```

5. 运行
```bash
python3 -m lite_synphonia \
  --seconds 20 \
  --transcription-provider deepgram \
  --summary-provider minimax \
  --embedding-provider minimax-embed \
  --pdf-path /你的/slide.pdf \
  --output-dir ./lite_synphonia_output
```

补充：
- 如果不做 PDF 匹配，去掉 `--pdf-path` 和 `--embedding-provider` 相关参数。
- provider 会保存到 `~/.config/lite_synphonia/providers.json`。
