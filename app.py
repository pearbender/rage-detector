from threading import Lock
from flask import Flask, render_template
from flask_socketio import SocketIO
import numpy as np
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import torch
import threading
from transformers import AutoTokenizer, AutoModelForSequenceClassification, LukeConfig
from twitchrealtimehandler import TwitchAudioGrabber

# Set this variable to "threading", "eventlet" or "gevent" to test the
# different async modes, or leave it set to None for the application to choose
# the best option based on installed packages.
async_mode = None

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode=async_mode)
thread = None
thread_lock = Lock()
text = ''

recorded = 0
transcribed = 0
emotion_names_jp = ['喜び', '悲しみ', '期待', '驚き', '怒り', '恐れ', '嫌悪', '信頼']
num_labels = len(emotion_names_jp)
prob = np.zeros(len(emotion_names_jp))

def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def record_audio():
    audio_grabber = TwitchAudioGrabber(
        twitch_url="https://www.twitch.tv/pearbender",
        blocking=True,
        segment_length=7,
        rate=44100,
        channels=2,
        dtype=np.int16
    )

    global recorded
    while True:
        print('Recording...')
        audio_data = audio_grabber.grab()
        if audio_data is not None:
            write('test.wav', 44100, audio_data)
            recorded += 1

def transcribe_audio():
    whisper_options = {
        'language': 'ja',
        'initial_prompt': 'うわああ、こいつマジでうぜぇ。むかつく本当に',
        'beam_size': 3,
        'best_of': 3,
        'temperature': (-2.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        'suppress_tokens': [],
        'vad_filter': True,
        'vad_parameters': dict(min_silence_duration_ms=800)
    }
    whisper_model = WhisperModel(
        'large-v2', device="cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime")
    config = LukeConfig.from_pretrained('Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime', output_hidden_states=True)
    model = AutoModelForSequenceClassification.from_pretrained('Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime', config=config)

    global recorded, transcribed, prob, text
    while True:
        if recorded <= transcribed:
            continue
        print('Transcribing...')
        segments, info = whisper_model.transcribe(
            'test.wav', **whisper_options)
        transcribed = recorded
        text = ''.join(segment.text for segment in segments)
        print(text)
        tokens = tokenizer(text, truncation=True, return_tensors="pt")
        tokens.to(model.device)
        preds = model(**tokens)
        prob = np_softmax(preds.logits.cpu().detach().numpy()[0])

def background_thread():
    """Example of how to send server generated events to clients."""
    while True:
        socketio.sleep(1)
        data = {
            'labels': emotion_names_jp,
            'values': prob.tolist(),
            'text': text

        }
        socketio.emit('my_response',
                      data)


@app.route('/')
def index():
    return render_template('index.html', async_mode=socketio.async_mode)


@socketio.event
def connect():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(background_thread)


if __name__ == '__main__':
    recording_thread = threading.Thread(target=record_audio)
    recording_thread.daemon = True
    recording_thread.start()

    transcribing_thread = threading.Thread(target=transcribe_audio)
    transcribing_thread.daemon = True
    transcribing_thread.start()

    socketio.run(app)
