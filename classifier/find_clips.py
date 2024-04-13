from pydub import AudioSegment, silence
import os
from faster_whisper import WhisperModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, LukeConfig
import numpy as np


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


def find_peak_amplitude_offset(segment):
    samples = np.array(segment.get_array_of_samples())
    i = np.argmax(samples)
    return int(round(i * 1000 / segment.frame_rate))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    whisper_model = WhisperModel('large-v2', device=device)

    tokenizer = AutoTokenizer.from_pretrained(
        "Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime")
    config = LukeConfig.from_pretrained(
        'Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime', output_hidden_states=True)
    sa_model = AutoModelForSequenceClassification.from_pretrained(
        'Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime', config=config)

    temp_file = 'temp.wav'
    vods_directory_path = 'data/vods-audio'
    file_paths = []
    for filename in os.listdir(vods_directory_path):
        if not filename.endswith('.mp3'):
            continue
        file_path = os.path.join(vods_directory_path, filename)
        file_paths.append(file_path)

    for i, file_path in enumerate(file_paths):
        base_name = os.path.basename(file_path)
        base_name_without_extension = os.path.splitext(base_name)[0]
        audio = AudioSegment.from_mp3(file_path)
        nonsilent_sections = silence.detect_nonsilent(
            audio, min_silence_len, -45, 100)
        new_nonsilent_sections = []
        maximum_section_length = 7000
        angry_clip_count = 0

        for [nonsilent_start, nonsilent_end] in nonsilent_sections:
            while nonsilent_end - nonsilent_start > maximum_section_length:
                new_nonsilent_sections.append(
                    [nonsilent_start, nonsilent_start + maximum_section_length])
                nonsilent_start += maximum_section_length
            if nonsilent_start + maximum_section_length <= len(audio):
                new_nonsilent_sections.append(
                    [nonsilent_start, nonsilent_start + maximum_section_length])
        nonsilent_sections = new_nonsilent_sections

        for [nonsilent_start, nonsilent_end] in nonsilent_sections:
            print(f"{file_path} ({i + 1}/{len(file_paths)
                                          }): {round(100 * nonsilent_start / len(audio))}%", end='\r')

            audio_segment = audio[nonsilent_start:nonsilent_end]

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

            if angry_prob < 0.1:
                continue

            peak_amplitude_offset = find_peak_amplitude_offset(audio_segment)
            clip_start = max(0, nonsilent_start + peak_amplitude_offset - 1000)
            clip_end = min(len(audio), nonsilent_start +
                           peak_amplitude_offset + 1000)
            clipped_audio_segment = audio[clip_start:clip_end]
            clipped_audio_segment.export(
                f"data/maybe-angry/{base_name_without_extension}_{clip_start}.wav", 'wav')
            angry_clip_count += 1

        print('')

    if os.path.exists(temp_file):
        os.remove(temp_file)


if __name__ == "__main__":
    main()
