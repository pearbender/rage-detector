
import os
from pydub import AudioSegment
import random
import sys


def main():
    vods_directory_path = 'data/vods-audio'
    file_paths = []
    random_clips_per_vod = int(sys.argv[1])
    clip_length = 2000

    for filename in os.listdir(vods_directory_path):
        if not filename.endswith('.mp3'):
            continue
        file_path = os.path.join(vods_directory_path, filename)
        file_paths.append(file_path)

    os.makedirs('data/random', exist_ok=True)

    for file_path in file_paths:
        base_name = os.path.basename(file_path)
        base_name_without_extension = os.path.splitext(base_name)[0]
        audio = AudioSegment.from_mp3(file_path)
        random_start_times = [random.randint(clip_length, len(
            audio) - clip_length) for _ in range(random_clips_per_vod)]
        for clip_start in random_start_times:
            clip_end = clip_start + clip_length
            audio_segment = audio[clip_start:clip_end]
            output_path = f"data/random/{
                base_name_without_extension}_{clip_start}.wav"
            print(output_path)
            audio_segment.export(output_path, "wav")


if __name__ == "__main__":
    main()
