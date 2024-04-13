import glob
import os
import pygame
import shutil

maybe_angry_directory = "data/maybe-angry"
angry_directory = "data/angry"
not_angry_directory = "data/not-angry"
file_paths = glob.glob(os.path.join(maybe_angry_directory, '*.wav'))

os.makedirs('data/angry', exist_ok=True)
os.makedirs('data/not-angry', exist_ok=True)

pygame.mixer.init()

for file_path in file_paths:
    file_name = os.path.basename(file_path)
    decided = False
    while not decided:
        print(file_path)
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pass
        pygame.mixer.music.unload()
        response = input(
            f"Is it angry ({file_path})? (y/n): ").strip().lower()
        if response == "y":
            destination_file_path = os.path.join(angry_directory, file_name)
            print(f"Moving {file_path} to {destination_file_path}...")
            shutil.move(file_path, destination_file_path)
            decided = True
        elif response == "n":
            destination_file_path = os.path.join(
                not_angry_directory, file_name)
            print(f"Moving {file_path} to {destination_file_path}...")
            shutil.move(file_path, destination_file_path)
            decided = True
