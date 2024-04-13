
from pydub import AudioSegment
import os
import torch
import os
import torch
import torchaudio
from torchaudio import transforms
import torch.nn.functional as F
import numpy as np
from model import *


def find_peak_amplitude_offset(segment):
    samples = np.array(segment.get_array_of_samples())
    i = np.argmax(samples)
    return int(round(i * 1000 / segment.frame_rate))


def main():
    model = AudioClassifier()
    model.load_state_dict(torch.load("./model.pt"))
    model.eval()

    temp_file = 'temp.wav'
    vods_directory_path = 'data/vods-audio'
    angry_directory = "data/angry"
    maybe_angry_directory = "data/maybe-angry"
    not_angry_directory = 'data/not-angry'
    file_paths = []
    clip_length = 2000
    for filename in os.listdir(vods_directory_path):
        if not filename.endswith('.mp3'):
            continue
        file_path = os.path.join(vods_directory_path, filename)
        file_paths.append(file_path)

    for i, file_path in enumerate(file_paths):
        base_name = os.path.basename(file_path)
        base_name_without_extension = os.path.splitext(base_name)[0]
        audio = AudioSegment.from_mp3(file_path)
        for clip_start in range(0, len(audio) - clip_length // 2, clip_length // 2):
            clip_end = clip_start + clip_length
            audio_segment = audio[clip_start:clip_end]

            if os.path.exists(temp_file):
                os.remove(temp_file)

            audio_segment.export(temp_file, 'wav')
            sig, sr = torchaudio.load(temp_file)
            spec = transforms.MelSpectrogram(
                sr, n_fft=1024, hop_length=None, n_mels=64)(sig)
            spec = transforms.AmplitudeToDB(top_db=80)(spec)
            spec = spec.unsqueeze(0)
            output = model.forward(spec)
            normalized_probs = F.softmax(output, dim=1)
            max_prob, class_index = torch.max(normalized_probs, dim=1)
            confidence = max_prob.item()
            predicted_class = class_index.item()
            is_angry = predicted_class == 0

            percent = clip_start / len(audio) * 100
            path = ''
            if is_angry and confidence >= 0.9:
                peak_amplitude_offset = find_peak_amplitude_offset(
                    audio_segment)
                new_clip_start = max(
                    0, clip_start + peak_amplitude_offset - clip_length // 2)
                new_clip_end = min(len(audio), clip_start +
                                   peak_amplitude_offset + clip_length // 2)
                clipped_audio_segment = audio[new_clip_start:new_clip_end]
                file_name = f"{base_name_without_extension}_{
                    new_clip_start}.wav"
                angry_path = os.path.join(angry_directory, file_name)
                not_angry_path = os.path.join(not_angry_directory, file_name)
                if not os.path.exists(angry_path) and not os.path.exists(not_angry_path):
                    maybe_angry_path = os.path.join(
                        maybe_angry_directory, file_name)
                    clipped_audio_segment.export(maybe_angry_path, 'wav')
                    path = maybe_angry_path
            print(f"{file_path} ({i + 1}/{len(file_paths)
                                          }) {percent:.0f}% {path}", end='\r')

    if os.path.exists(temp_file):
        os.remove(temp_file)


if __name__ == "__main__":
    main()
