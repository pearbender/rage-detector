import asyncio
import websockets
from faster_whisper import WhisperModel
from scipy.signal import resample
import numpy as np
import functools
from transformers import AutoTokenizer, AutoModelForSequenceClassification, LukeConfig
import time
import torch
import torchaudio
from torchaudio import transforms
import torch.nn.functional as F
from scipy.io.wavfile import write
import json

from model import *

busy = False


async def hello(websocket, path, whisper_options, whisper_model, wrime_tokenizer, wrime_model, classifier_model):
    global busy

    if busy:
        return

    busy = True

    print("Starting ffmpeg process...")

    proc = await asyncio.create_subprocess_shell(
        'ffmpeg -f webm -i pipe: -f f32le -ac 1 -',
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL
    )

    print("Starting worker task...")

    worker = asyncio.create_task(worker_task(
        websocket, proc, whisper_options, whisper_model, wrime_tokenizer, wrime_model, classifier_model))

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


def prepare_file(audio_file):
    sig, sr = torchaudio.load(audio_file)
    spec = transforms.MelSpectrogram(
        sr, n_fft=1024, hop_length=None, n_mels=64)(sig)
    spec = transforms.AmplitudeToDB(top_db=80)(spec)

    return spec.unsqueeze(0)


def get_prediction(model, audio_file):
    inputs = prepare_file(audio_file)
    output = model.forward(inputs)
    normalized_probs = F.softmax(output, dim=1)
    max_prob, class_index = torch.max(normalized_probs, dim=1)
    confidence = max_prob.item()
    predicted_class = class_index.item()
    return confidence, predicted_class


def is_angry(model, audio_file):
    _, angry = get_prediction(model, audio_file)
    return angry == 0


async def worker_task(websocket, proc, whisper_options, whisper_model, wrime_tokenizer, wrime_model, classifier_model):
    print("Worker task started.")

    chunks = b''
    sample_rate = 48000
    target_sample_rate = 16000
    resample_ratio = target_sample_rate / sample_rate
    seconds = 4
    overlap_coefficient = 2
    read_chunk_len = int((seconds / overlap_coefficient) * 4 * sample_rate)
    process_chunk_len = read_chunk_len * overlap_coefficient
    device = "cpu"

    while True:
        read = 0
        while read < read_chunk_len:
            chunk = await proc.stdout.read(read_chunk_len - read)
            chunks += chunk
            read += len(chunk)

        if len(chunks) < process_chunk_len:
            continue

        start_time = time.time()

        samples = np.frombuffer(chunks, dtype=np.float32)
        second_half_samples = samples[(len(samples) // 2):]
        scaled_samples = np.int16(second_half_samples * 32767)
        write('temp.wav', sample_rate, scaled_samples)

        confidence, predicted_class = get_prediction(
            classifier_model, 'temp.wav')

        samples = resample(samples, int(len(samples) * resample_ratio))
        samples = samples.astype(np.float32)

        segments, _ = whisper_model.transcribe(samples, **whisper_options)
        text = ''.join(segment.text for segment in segments)

        inputs = wrime_tokenizer(text, truncation=True,
                                 return_tensors="pt").to(device)
        wrime_model = wrime_model.to(device)
        outputs = wrime_model(**inputs)
        prob = np_softmax(outputs.logits.cpu().detach().numpy()[0])

        rounded_prob = round(prob[4] * 20) / 20
        percent = int(round(prob[4] * 100))
        num_bars = int(rounded_prob * 20)
        bars = f"[{'|' * num_bars}{' ' * (20 - num_bars)}]"

        end_time = time.time()
        elapsed_time = end_time - start_time
        realtime_ratio = elapsed_time / (seconds / overlap_coefficient)

        print(f"{realtime_ratio:.1f} {percent: >3}% {
              bars} {confidence:.2f} {predicted_class} {text}")
        data = {
            'p': str(prob[4]),
            'predicted_class': str(predicted_class),
            'confidence': str(confidence)
        }
        await websocket.send(json.dumps(data))

        chunks = chunks[read_chunk_len:]


async def main():
    whisper_options = {
        'language': 'ja',
        'beam_size': 5,
        'best_of': 1,
        'temperature': 0,
        'suppress_tokens': [],
        'vad_filter': True,
        'vad_parameters': dict(min_silence_duration_ms=800)
    }

    print("Loading whisper model...")

    whisper_model = WhisperModel(
        'base', device="cpu", cpu_threads=8, compute_type="int8")

    print("Loading sentiment analysis model...")

    wrime_tokenizer = AutoTokenizer.from_pretrained(
        "Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime")
    wrime_config = LukeConfig.from_pretrained(
        'Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime', output_hidden_states=True)
    wrime_model = AutoModelForSequenceClassification.from_pretrained(
        'Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime', config=wrime_config)

    print("Loading audio classifier model...")

    classifier_model = AudioClassifier()
    classifier_model.load_state_dict(torch.load("./model.pt"))
    classifier_model.eval()

    print("Starting WebSocket server...")

    async with websockets.serve(
            functools.partial(hello,
                              whisper_options=whisper_options,
                              whisper_model=whisper_model,
                              wrime_tokenizer=wrime_tokenizer,
                              wrime_model=wrime_model,
                              classifier_model=classifier_model),
            "0.0.0.0", 8765):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
