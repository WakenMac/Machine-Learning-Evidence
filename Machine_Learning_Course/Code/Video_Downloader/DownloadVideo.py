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
ydl_opts = {
    'noplaylist': True, # Only download single video, not a playlist
    'windowsfilenames':True,
    'paths': {"home": r"C:\\Users\\Waks\\Downloads\\USEP BSCS\\School Work\\BSCS 3 - 1st Sem\\CSDS 314 - Machine Learning\\LE\\Machine-Learning-Evidence\\Machine_Learning_Course\\Piano_Recordings"},
    'outtmpl': '%(title)s.%(ext)s', # Output filename template
    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
    "quiet":True
}

# Create a YoutubeDL object with the defined options
ydl = YoutubeDL(ydl_opts)

# Download a video
# ydl.download(['https://youtu.be/YMO97BPklrs?list=RDYMO97BPklrs'])

file = open('C:\\Users\\Waks\\Downloads\\USEP BSCS\\School Work\\BSCS 3 - 1st Sem\\CSDS 314 - Machine Learning\\LE\\Machine-Learning-Evidence\\Machine_Learning_Course\\Code\\Video_Downloader\\Waks_YT_Links.txt', 'r')

urls = [line.strip() for line in file]

for url in urls:
    # Or extract information without downloading
    info_dict = ydl.extract_info(url, download=False)
    print(f'Downloaded: {info_dict.get("title")} Duration: {info_dict.get("duration")} secs.')