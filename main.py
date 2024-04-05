import asyncio
import ssl
import websockets
from faster_whisper import WhisperModel
import torch
from scipy.signal import resample
import numpy as np
import functools
from transformers import AutoTokenizer, AutoModelForSequenceClassification, LukeConfig


busy = False


async def hello(websocket, path, whisper_options, whisper_model, tokenizer, model):
    global busy

    if busy:
        return

    busy = True

    print("Starting ffmpeg process...")

    proc = await asyncio.create_subprocess_shell(
        'ffmpeg -f webm -i pipe: -f f32le -',
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL
    )

    print("Starting worker task...")

    worker = asyncio.create_task(worker_task(
        websocket, proc, whisper_options, whisper_model, tokenizer, model))

    try:
        async for chunk in websocket:
            proc.stdin.write(chunk)
            await proc.stdin.drain()
    finally:
        print("Making server available to new connections...")

        worker.cancel()
        proc.kill()
        busy = False

        print("Ready for new connections.")


def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


async def worker_task(websocket, proc, whisper_options, whisper_model, tokenizer, model):
    print("Worker task started.")

    chunks = b''
    sample_rate = 48000
    target_sample_rate = 16000
    resample_ratio = target_sample_rate / sample_rate
    min_chunk_len = 7 * 4 * sample_rate
    device = "cpu"

    while True:
        while len(chunks) < min_chunk_len:
            chunk = await proc.stdout.read(min_chunk_len - len(chunks))
            chunks += chunk

        samples = np.frombuffer(chunks, dtype=np.float32)
        samples = resample(samples, int(len(samples) * resample_ratio))
        samples = samples.astype(np.float32)

        segments, _ = whisper_model.transcribe(samples, **whisper_options)
        text = ''.join(segment.text for segment in segments)

        inputs = tokenizer(text, truncation=True,
                           return_tensors="pt").to(device)
        model = model.to(device)
        outputs = model(**inputs)
        prob = np_softmax(outputs.logits.cpu().detach().numpy()[0])

        rounded_prob = round(prob[4] * 20) / 20
        num_bars = int(rounded_prob * 20)
        bars = f"[{'|' * num_bars}{' ' * (20 - num_bars)}]"

        print(f"{bars} {text}")
        await websocket.send(f"{prob[4]}")

        chunks = b''


async def main():
    whisper_options = {
        'language': 'ja',
        'beam_size': 3,
        'best_of': 3,
        'temperature': (-2.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        'suppress_tokens': [],
        'vad_filter': True,
        'vad_parameters': dict(min_silence_duration_ms=800)
    }

    print("Loading whisper model...")

    whisper_model = WhisperModel(
        'tiny', device="cpu", compute_type="int8")

    print("Loading sentiment analysis model...")

    tokenizer = AutoTokenizer.from_pretrained(
        "Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime")
    config = LukeConfig.from_pretrained(
        'Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime', output_hidden_states=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        'Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime', config=config)

    print("Starting WebSocket server...")

    async with websockets.serve(
            functools.partial(hello,
                              whisper_options=whisper_options,
                              whisper_model=whisper_model,
                              tokenizer=tokenizer,
                              model=model),
            "0.0.0.0", 8765):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
