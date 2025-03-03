import os
import subprocess

input_dir = "I:\\mar2nd" # my samsung usb c drive
output_dir = "extracted_frames"  # save frames onto local storage

os.makedirs(output_dir, exist_ok=True)

mov_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".mov")] # only process .mov files since thats what my camera records to @ 1080p 60fps

for mov_file in mov_files:
    video_path = os.path.join(input_dir, mov_file)
    video_name = os.path.splitext(mov_file)[0]
    video_output_dir = os.path.join(output_dir, video_name)

    os.makedirs(video_output_dir, exist_ok=True)

    output_pattern = os.path.join(video_output_dir, "frame_%04d.png")
    ffmpeg_command = [ # YOU NEED FFMPEG INSTALLED ON YOUR MACHINE FOR THIS, IF YOU NEED IT ON WINDOWS USE <choco install ffmpeg>, macOS USE <brew install ffmpeg>
        "ffmpeg",
        "-i", video_path,       # Full path to input file
        "-vf", "fps=2",       # 1 frame every 0.5 seconds
        "-q:v", "2",            # High quality
        output_pattern          # Output pattern
    ]

    print(f"Extracting frames from {mov_file}...")
    subprocess.run(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("Frame extraction complete! Check the 'extracted_frames' folder on your Samsung USB.")
