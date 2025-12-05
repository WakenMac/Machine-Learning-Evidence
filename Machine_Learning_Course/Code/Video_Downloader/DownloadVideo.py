# Author: Waken Cean C. Maclang
# Date Last Edited: November 1, 2025
# Course: Machine Learning
# Task: Learning Evidence

# DownloadVideo.py 
#     A file that tracks and downloads respective youtube videos for our dataset

# Works with Python 3.10.0

# Install yt_dlp before running the code.
from yt_dlp import YoutubeDL

# Define options for the download
# For Video downloads
ydl_opts = {
    'noplaylist': True, # Only download single video, not a playlist
    'windowsfilenames':True,
    'paths': {"home": r"Machine_Learning_Course\\Piano_Recordings"},
    'outtmpl': '%(title)s.%(ext)s', # Output filename template
    'format': 'bestvideo[height<=720][height>=480][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
    'merge_output_format':"mp4",
    "quiet":True
}

# For Audio Downloads
# ydl_opts = {
#     'noplaylist': True,  # Don't process playlists
#     'windowsfilenames': True,
#     'paths': {'home': r'C:\\Users\\Waks\\Music\\Downloaded Music'},
#     'outtmpl': '%(title)s.%(ext)s',  # Output filename template
#     'format': 'bestaudio/best',  # Download the best available audio
#     'postprocessors': [{
#         'key': 'FFmpegExtractAudio',
#         'preferredcodec': 'mp3',
#         'preferredquality': '0',  # 0 = best quality
#     }],
#     'quiet': True
# }

# Create a YoutubeDL object with the defined options
ydl = YoutubeDL(ydl_opts)

# Download a video
# ydl.download(['https://www.youtube.com/watch?v=osy7gJBWNuA&list=RDosy7gJBWNuA&start_radio=1&pp=ygUMc2kgaGFuYXRzdWthoAcB'])

file = open('Machine_Learning_Course\\Code\\Video_Downloader\\Waks_YT_Links.txt', 'r')

urls = [[line.strip() for line in file][1]]

urls = ['https://www.youtube.com/watch?v=lv5xPlm6etI']

for url in urls:
    # Or extract information without downloading
    info_dict = ydl.extract_info(url, download=True)
    print(f'Downloaded: {info_dict.get("title")} Duration: {info_dict.get("duration")} secs.')
