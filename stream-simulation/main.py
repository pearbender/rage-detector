
from pydub import AudioSegment, silence
import sys
import os
from faster_whisper import WhisperModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, LukeConfig
import numpy as np
import json


min_silence_len = 800
whisper_options = {
    'language': 'ja',
    'initial_prompt': "PearBender welcome, fuck english... ええと こんばんは、Alex welcome, いやねぇ 今日はねぇ あ ところでさ ごめん あの 今日ねぇ 今日ねぇ みんな 今日ねぇ, cha- cha- what is it? cha- chazay? あとなんか変な味がする… 口の中, Pero welcome, Mathew welcome, Ender welcome. I'm playing wa- i don't get it wa- warhammer",
    'beam_size': 5,
    'best_of': 1,
    'temperature': 0,
    'suppress_tokens': [],
    'vad_filter': True,
    'vad_parameters': dict(min_silence_duration_ms=min_silence_len)
}


def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


def main():
    file_path = sys.argv[1]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    whisper_model = WhisperModel('large-v2', device=device)

    tokenizer = AutoTokenizer.from_pretrained(
        "Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime")
    config = LukeConfig.from_pretrained(
        'Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime', output_hidden_states=True)
    sa_model = AutoModelForSequenceClassification.from_pretrained(
        'Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime', config=config)

    temp_file = 'temp.wav'
    base_name = os.path.basename(file_path)
    base_name_without_extension = os.path.splitext(base_name)[0]
    audio = AudioSegment.from_mp3(file_path)
    audio = audio.set_channels(1)
    chunk_length = 7000
    half_chunk_length = 3500
    data = []

    for chunk_end in range(chunk_length, len(audio), half_chunk_length):
        chunk_start = chunk_end - chunk_length
        audio_segment = audio[chunk_start:chunk_end]

        if os.path.exists(temp_file):
            os.remove(temp_file)

        audio_segment.export(temp_file, 'wav')
        text_segments, _ = whisper_model.transcribe(
            temp_file, **whisper_options)
        text = ''.join(text_segment.text for text_segment in text_segments)
        text = text.replace('\n', ' ')
        text = text.strip()

        inputs = tokenizer(text, truncation=True,
                           return_tensors="pt").to(device)
        sa_model = sa_model.to(device)
        outputs = sa_model(**inputs)
        prob = np_softmax(outputs.logits.cpu().detach().numpy()[0])
        angry_prob = prob[4]
        data.append({
            "t": chunk_start,
            "p": angry_prob
        })
        percentage = (chunk_start / len(audio)) * 100
        print(f"{percentage:.0f}% ({chunk_start}/{len(audio)}) {angry_prob}", end="\r")

    if os.path.exists(temp_file):
        os.remove(temp_file)

    with open(f"{base_name_without_extension}.json", "w") as json_file:
        json.dump(data, json_file, indent=4)


if __name__ == "__main__":
    main()
