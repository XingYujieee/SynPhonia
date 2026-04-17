"""LiteSynphonia — zero-local-model edition of MergeSyn.

All AI operations (transcription, summarization, embeddings) are performed
via remote API calls.  No PyTorch, no Whisper binary, no local model files
are required.

Providers are managed through the shared provider registry:

    python3 -m lite_synphonia providers list
    python3 -m lite_synphonia providers add deepgram \\
        --base-url https://api.deepgram.com \\
        --model whisper-large --api-key <key> --service transcription
    python3 -m lite_synphonia providers add minimax \\
        --base-url https://api.minimaxi.chat/v1 \\
        --model MiniMax-Text-01 --api-key <key> --service summarization

Run the full pipeline:

    python3 -m lite_synphonia \\
        --seconds 60 \\
        --transcription-provider deepgram \\
        --summary-provider minimax \\
        --embedding-provider minimax \\
        --pdf-path slides.pdf \\
        --output-dir output/
"""

__version__ = "0.1.0"
