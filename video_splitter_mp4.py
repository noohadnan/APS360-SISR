import os
import subprocess

input_dir = "I:\\book_vids" 
output_dir = "I:\\book_vids\\extracted_book_frames"

os.makedirs(output_dir, exist_ok=True)

mp4_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".mp4")]

for mp4_file in mp4_files:
    video_path = os.path.join(input_dir, mp4_file)
    video_name = os.path.splitext(mp4_file)[0]
    video_output_dir = os.path.join(output_dir, video_name)

    os.makedirs(video_output_dir, exist_ok=True)

    output_pattern = os.path.join(video_output_dir, "frame_%04d.png")
    ffmpeg_command = [ # YOU NEED FFMPEG INSTALLED ON YOUR MACHINE FOR THIS, IF YOU NEED IT ON WINDOWS USE <choco install ffmpeg>, macOS USE <brew install ffmpeg>
        "ffmpeg",
        "-i", video_path,       # Full path to input file
        "-vf", "fps=1",       # 1 frame every second
        "-q:v", "2",            # High quality
        output_pattern          # Output pattern
    ]

    print(f"Extracting frames from {mp4_file}...")
    subprocess.run(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("Frame extraction complete! Check the 'extracted_book_frames' folder on Samsung USB.")
