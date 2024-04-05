import asyncio
import ssl
import websockets
from faster_whisper import WhisperModel
import torch
from scipy.signal import resample
import numpy as np
import functools
from transformers import AutoTokenizer, AutoModelForSequenceClassification, LukeConfig


async def hello(websocket, path, proc, whisper_options, whisper_model, tokenizer, model):
    print("Starting worker task...")
    asyncio.create_task(worker_task(
        websocket, proc, whisper_options, whisper_model, tokenizer, model))

    while True:
        chunk = await websocket.recv()
        proc.stdin.write(chunk)
        await proc.stdin.drain()


def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


async def worker_task(websocket, proc, whisper_options, whisper_model, tokenizer, model):
    print("Worker task started.")

    chunks = b''
    sample_rate = 48000
    target_sample_rate = 16000
    resample_ratio = target_sample_rate / sample_rate
    min_chunk_len = 5 * 4 * sample_rate

    while True:
        while len(chunks) < min_chunk_len:
            chunk = await proc.stdout.read(min_chunk_len - len(chunks))
            chunks += chunk

        samples = np.frombuffer(chunks, dtype=np.float32)
        samples = resample(samples, int(len(samples) * resample_ratio))
        samples = samples.astype(np.float32)

        segments, _ = whisper_model.transcribe(samples, **whisper_options)
        text = ''.join(segment.text for segment in segments)

        tokens = tokenizer(text, truncation=True, return_tensors="pt")
        tokens.to(model.device)
        preds = model(**tokens)
        prob = np_softmax(preds.logits.cpu().detach().numpy()[0])

        print(prob[4], text)
        await websocket.send(f"{prob[4]}")

        chunks = b''


async def main():
    whisper_options = {
        'language': 'ja',
        'initial_prompt': 'うわああ、こいつマジでうぜぇ。ムカつく本当に',
        'beam_size': 3,
        'best_of': 3,
        'temperature': (-2.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        'suppress_tokens': [],
        'vad_filter': True,
        'vad_parameters': dict(min_silence_duration_ms=800)
    }

    print("Loading whisper model...")

    whisper_model = WhisperModel(
        'large-v2', device="cuda" if torch.cuda.is_available() else "cpu")

    print("Loading sentiment analysis model...")

    tokenizer = AutoTokenizer.from_pretrained(
        "Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime")
    config = LukeConfig.from_pretrained(
        'Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime', output_hidden_states=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        'Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime', config=config)

    print("Starting ffmpeg process...")

    proc = await asyncio.create_subprocess_shell(
        'ffmpeg -f webm -i pipe: -f f32le -',
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL
    )

    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain("cert.pem", "cert.key")

    print("Starting WebSocket server...")

    async with websockets.serve(
            functools.partial(hello,
                              proc=proc,
                              whisper_options=whisper_options,
                              whisper_model=whisper_model,
                              tokenizer=tokenizer,
                              model=model),
            "localhost", 8765, ssl=ssl_context):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
