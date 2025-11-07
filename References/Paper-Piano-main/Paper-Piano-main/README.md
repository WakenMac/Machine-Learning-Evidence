# Paper Piano

**Paper Piano** is an interactive application that lets you “play” a printed piano using green markers and your computer’s camera. It combines OpenCV for computer vision, MediaPipe Hands for finger tracking, and Pygame for sound playback. Perfect for demos, tangible interface prototypes, and digital art projects.

<p align="center">
  <img src="https://github.com/user-attachments/assets/a9323132-076e-4058-a5cf-1e37fcf6be49"
alt="Paper Piano Template" width="70%" />
</p>

---

## 📖 Overview

1. **Green Marker Detection**  
   - Fine-tune detection with an **HSV Adjust** window featuring six sliders (Hmin/Smin/Vmin | Hmax/Smax/Vmax).  
   - Three built-in presets: Dark Green, Medium Green, and Light Green.

2. **Perspective Transformation**  
   - Click four corners of your printed template to correct the perspective and map the piano in the video feed.

3. **Interaction Modes**  
   - **Dynamic**: continuously redetects marker positions—ideal if you move the sheet.  
   - **Static**: once 12 note markers + 2 control markers are found, their positions freeze and remain constant each frame.

4. **Note Playback**  
   - Sound files located in `Piano_Samples/`:  
     ```
     Piano_Samples/
     ├── C0.mp3 … C7.mp3
     ├── Db0.mp3 … Db7.mp3
     └── … up to B7.mp3
     ```
   - When a tracked fingertip overlaps a note marker, that note plays immediately.

5. **Octave Control**  
   - On-screen buttons (or keyboard shortcuts):  
     - Oct– / Oct+   
     - Toggle between 1-octave (12 notes) and 2-octave (24 notes) layouts.  
   - Two large control markers: first = Oct–, second = Oct+. Hover for 0.5 s “cooldown” to switch.

6. **UI and Keyboard Shortcuts**  
   | Button / Key | Action                        |  
   |--------------|-------------------------------|  
   | Rst / r      | Reset perspective points      |  
   | Togl / t     | Toggle static/dynamic mode    |  
   | Frnt / f     | Toggle frontal/normal view    |  
   | o / p        | Octave Down / Octave Up       |  
   | g            | Rotate 180°                   |  
   | m            | Mirror horizontally           |  
   | n            | Toggle 1 ↔ 2 octaves          |  
   | h / j / k    | Apply HSV presets             |  
   | q            | Quit                          | 

---

## 📂 Project Structure

```
Paper_Piano/
├── main.py                          # Main application script
├── Piano_Samples/                   # MP3 sound files for each note
│   ├── C0.mp3 … B0.mp3
│   ├── C1.mp3 … B1.mp3
│   └── … up to C7.mp3 … B7.mp3
├── piano_imgs/                      # Template images
│   ├── Paper Piano_2.jpg            # Template with green markers
│   ├── Paper Piano_3.jpg            # Alternative template
│   └── Paper Piano.docx             # Template images sheet
├── Paper Piano (2 Octaves).ipynb
├── Paper Piano (2 Octaves)-Change Octave Buttons.ipynb
└── README.md                        # This file
```

---

## 🚀 Installation & Running

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Eliasjapi-dev/Paper-Piano.git
   cd Paper_Piano
   ```

2. **(Optional) Create a virtual environment**  
   ```bash
   python -m venv venv
   source venv/bin/activate     # Linux/macOS
   venv\\Scripts\\activate      # Windows
   ```

3. **Install dependencies**  
   ```bash
   pip install opencv-python mediapipe pygame numpy
   ```

4. **Run the application**  
   ```bash
   python main.py
   ```

---

## 🎥 Demo Videos

<p align="center">
  <a href="https://youtu.be/x4NhaETVLHE">
    <img src="https://github.com/user-attachments/assets/3fa8fb33-9a5e-42e2-ba6a-8fae8053bd26" 
      alt="Step-by-Step Guide" width="45%" />
  </a>
  <a href="https://youtu.be/RmHnPL2CEOw">
    <img src="https://github.com/user-attachments/assets/7b25d7af-780a-49dd-83c2-c6cbd7bc4650"
      alt="Full Demo" width="45%" />
  </a>
</p>

<p align="center">
  <a href="https://youtu.be/x4NhaETVLHE"><strong>▶️ Watch Setup & Calibration Guide</strong></a> &nbsp;|&nbsp;  
  <a href="https://youtu.be/RmHnPL2CEOw"><strong>▶️ Watch Full Demonstration</strong></a>
</p>

---

## 🤝 Contributing

1. Fork the repository.  
2. Create a branch: `git checkout -b feature/new-feature`.  
3. Commit your changes: `git commit -m "Add new feature"`.  
4. Open a pull request.

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
