# SEPIA - Air Piano

**Transform your hands into a virtual piano with AI-powered gesture recognition**

SEPIA (Smart Expression Piano with Intelligent Air-control) is an innovative air piano system that lets you play chords and melodies by simply raising your fingers in front of a camera. Using computer vision and hand tracking, it maps your 10 fingers to musical chords across multiple genres.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

- **10-Finger Chord System**: Each finger (5 per hand) triggers a different chord
- **8 Musical Presets**: Pop, Jazz, Blues, Rock, Classical, Gospel, Ambient, and Single Notes
- **Real-time Hand Tracking**: Accurate finger detection with improved thumb recognition
- **Visual Feedback**: 
  - On-screen piano keys showing active chords
  - Particle effects when notes are played
  - Finger indicators with color coding
  - Recent chord history display
- **Comprehensive Chord Library**: 80+ chords including major, minor, 7th, maj7, sus2, sus4, diminished, augmented, and 9th chords
- **MIDI Output**: Generate real MIDI notes for recording or external synthesis
- **Smooth Performance**: Optimized for 60 FPS with state smoothing

## Chord Presets

### Pop
Popular chord progressions (I-V-vi-IV): C, G, Am, F, Dm, Em, C7, G7, Am7, Fmaj7

### Jazz
Jazz progressions (ii-V-I): Dm7, G7, Cmaj7, Am7, Em7, Fmaj7, Bm7, E7, A7, D7

### Blues
12-bar blues: C7, F7, G7, C9, F9, G9, Dm7, Em7, Am7, Gm7

### Rock
Rock progressions and power chords: C, F, G, Am, Em, Dm, Bb, Csus4, Fsus2, Gsus4

### Classical
Classical progressions: C, Dm, Em, F, G, Am, Bdim, Cmaj7, Fmaj7, G7

### Gospel
Gospel progressions: Cmaj7, Fmaj7, Gmaj7, Am7, Dm7, Em7, C9, F9, G9, Bm7

### Ambient
Atmospheric chords: Csus2, Gsus2, Fsus2, Asus2, Dsus2, Esus2, Cmaj7, Gmaj7, Fmaj7, Amaj7

### Single Notes
Original single-note mode: C4 through E5

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam
- MIDI synthesizer (built-in or external)

### Install Dependencies

```bash
pip install opencv-python
pip install cvzone
pip install pygame
pip install mediapipe
pip install numpy
```

### Quick Start

```bash
# Clone the repository
git clone https://github.com/MohsinCell/SEPIA-air-piano.git
cd SEPIA-air-piano

# Run the application
python sepia.py
```

## How to Use

### Basic Controls

1. **Position yourself** in front of the camera with good lighting
2. **Show both hands** to the camera
3. **Raise fingers** to play chords
4. **Lower fingers** to stop the chord (auto-sustain for 0.5 seconds)

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `1-8` | Switch between presets |
| `L` | Toggle landmark visualization |
| `R` | Reset all active notes |
| `Q` | Quit application |

### Finger Mapping

**Left Hand** (Left side of screen):
- Thumb → Chord 1
- Index → Chord 2
- Middle → Chord 3
- Ring → Chord 4
- Pinky → Chord 5

**Right Hand** (Right side of screen):
- Thumb → Chord 6
- Index → Chord 7
- Middle → Chord 8
- Ring → Chord 9
- Pinky → Chord 10

## User Interface

The interface displays:
- **Top Left**: Title, current preset, active chords, FPS
- **Top Right**: Recent chord history (last 7 chords)
- **Bottom**: Visual piano keyboard with chord names
- **Finger Tips**: Color-coded circles when fingers are raised
- **Particle Effects**: Visual feedback when chords are played

## Configuration

You can customize the following parameters in the code:

```python
# Sustain time for notes (seconds)
SUSTAIN_TIME = 0.5

# MIDI velocity (0-127)
NOTE_VELOCITY = 100

# Show hand landmarks
SHOW_LANDMARKS = True

# Camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Detection confidence (0.0-1.0)
detector = HandDetector(detectionCon=0.8, minTrackCon=0.6)
```

## Technical Details

### Hand Detection
- Uses MediaPipe hand tracking via CVZone
- Improved thumb detection algorithm for better accuracy
- State smoothing to reduce jitter
- Supports up to 2 hands simultaneously

### Audio Engine
- Pygame MIDI for sound generation
- Real-time note triggering and release
- Multi-threaded note sustain management
- Chord playback with simultaneous notes

### Performance Optimizations
- Circular buffer for state management
- Efficient particle effect system
- FPS smoothing and monitoring
- Optimized drawing routines

## Troubleshooting

### No sound output
- Check MIDI device settings in your system
- Verify Pygame MIDI initialization
- Try different MIDI instruments/channels

### Poor hand detection
- Ensure good lighting conditions
- Position hands 1-2 feet from camera
- Avoid cluttered backgrounds
- Adjust `detectionCon` parameter

### Low FPS
- Reduce camera resolution
- Disable landmark visualization (`L` key)
- Close other resource-intensive applications
- Check CPU usage

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Ideas for Contribution
- Add more chord presets
- Implement gesture-based effects (vibrato, pitch bend)
- Add recording and playback features
- Create a GUI for chord customization
- Support for drum sounds with gestures
- Multi-language support

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [CVZone](https://github.com/cvzone/cvzone) for hand tracking
- [MediaPipe](https://mediapipe.dev/) for hand landmark detection
- [Pygame](https://www.pygame.org/) for MIDI handling
- Inspired by air piano and virtual instrument projects

---

**Made with Python**
