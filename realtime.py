import pyaudio
import numpy as np
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import torch
import threading
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.backends.backend_qt5agg

recorded = 0
transcribed = 0
emotion_names_jp = ['喜び', '悲しみ', '期待', '驚き', '怒り', '恐れ', '嫌悪', '信頼']
num_labels = len(emotion_names_jp)
prob = np.zeros(len(emotion_names_jp))
plot_text = ''


def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


def record_audio():
    p = pyaudio.PyAudio()
    default_device_index = p.get_default_input_device_info()['index']
    stream = p.open(format=pyaudio.paInt16,
                    channels=2,
                    rate=44100,
                    input=True,
                    input_device_index=default_device_index,
                    frames_per_buffer=44100)

    global recorded
    while True:
        print('Recording...')
        block = stream.read(10 * 44100)
        audio_data = np.frombuffer(block, dtype=np.int16)
        audio_data = audio_data.reshape(-1, 2)
        write('test.wav', 44100, audio_data)
        recorded += 1


def transcribe_audio():
    whisper_options = {
        'language': 'ja',
        'initial_prompt': "PearBender welcome, fuck english... ええと こんばんは、Alex welcome, いやねぇ 今日はねぇ あ ところでさ ごめん あの 今日ねぇ 今日ねぇ みんな 今日ねぇ, cha- cha- what is it? cha- chazay? あとなんか変な味がする… 口の中, Pero welcome, Mathew welcome, Ender welcome. I'm playing wa- i don't get it wa- warhammer",
        'beam_size': 3,
        'best_of': 3,
        'temperature': (-2.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        'suppress_tokens': [],
        'vad_filter': True,
        'vad_parameters': dict(min_silence_duration_ms=800)
    }
    whisper_model = WhisperModel(
        'large-v2', device="cuda" if torch.cuda.is_available() else "cpu")

    tokenizer_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    checkpoint = 'test_trainer/checkpoint-2000'
    model = AutoModelForSequenceClassification.from_pretrained(

        checkpoint, num_labels=num_labels)

    global recorded, transcribed, prob, plot_text
    while True:
        if recorded <= transcribed:
            continue
        print('Transcribing...')
        segments, info = whisper_model.transcribe(
            'test.wav', **whisper_options)
        transcribed += 1
        text = ''.join(segment.text for segment in segments)
        print(text)
        model.eval()
        tokens = tokenizer(text, truncation=True, return_tensors="pt")
        tokens.to(model.device)
        preds = model(**tokens)
        prob = np_softmax(preds.logits.cpu().detach().numpy()[0])
        plot_text = text


out_dict = {n: p for n, p in zip(emotion_names_jp, prob)}
df = pd.DataFrame(out_dict.items(), columns=['name', 'prob'])
sns.set(font='Ms Gothic')
plt.figure(figsize=(8, 3))
barplot = sns.barplot(x='name', y='prob', data=df, hue='name')
barplot.set_ylim(0, 1)
plt.title('入力文 : ' + plot_text, fontsize=12)

recording_thread = threading.Thread(target=record_audio)
recording_thread.daemon = True
recording_thread.start()

transcribing_thread = threading.Thread(target=transcribe_audio)
transcribing_thread.daemon = True
transcribing_thread.start()

while plt.fignum_exists(1):
    for bar, h in zip(barplot.patches, prob):
        bar.set_height(h)
    plt.title('入力文 : ' + plot_text, fontsize=15)
    plt.draw()
    plt.pause(1)


exit(0)
