import os
from moviepy import VideoFileClip

# Define input and output directories: --->
video_dir = 'grid_videos'
audio_dir = 'audio'

# Create output directory if it doesn't exist: --->
os.makedirs(audio_dir, exist_ok=True)

# Process each .mpg video: --->
for filename in os.listdir(video_dir):
    if filename.endswith('.mpg'):
        video_path = os.path.join(video_dir, filename)
        base_name = os.path.splitext(filename)[0]
        audio_path = os.path.join(audio_dir, f'{base_name}.wav')

        print(f'Extracting audio from {filename}...')

        try:
            clip = VideoFileClip(video_path)
            clip.audio.write_audiofile(audio_path, codec='pcm_s16le')  # WAV format
            clip.close()
        except Exception as e:
            print(f'Error processing {filename}: {e}')

print("Audio extraction complete.")
