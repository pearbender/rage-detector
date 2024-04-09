import os
import subprocess

audio_dir = 'data/vods-audio'
if not os.path.exists(audio_dir):
    os.makedirs(audio_dir)

vods_dir = 'data/vods/sf6'
for vod in os.listdir(vods_dir):
    if not vod.endswith('.mkv'):
        continue
    vod_path = os.path.join(vods_dir, vod)
    audio = os.path.join(audio_dir, f'{os.path.splitext(vod)[0]}.mp3')
    subprocess.run(['ffmpeg', '-i', vod_path, '-q:a', '0', '-map', 'a', audio])