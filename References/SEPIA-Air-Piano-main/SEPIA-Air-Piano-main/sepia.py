import cv2
import threading
import pygame.midi
import time
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from collections import deque

# Initialize Pygame MIDI
pygame.midi.init()
player = pygame.midi.Output(0)
player.set_instrument(0)  # Acoustic Grand Piano

# Initialize Hand Detector 
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)

# Detector settings for thumb recognition
detector = HandDetector(detectionCon=0.8, minTrackCon=0.6, maxHands=2)

# Comprehensive Chord Library - All common chord types
CHORD_LIBRARY = {
    # Major Chords
    "C Major": [60, 64, 67],
    "D Major": [62, 66, 69],
    "E Major": [64, 68, 71],
    "F Major": [65, 69, 72],
    "G Major": [67, 71, 74],
    "A Major": [69, 73, 76],
    "B Major": [71, 75, 78],
    
    # Minor Chords
    "C Minor": [60, 63, 67],
    "D Minor": [62, 65, 69],
    "E Minor": [64, 67, 71],
    "F Minor": [65, 68, 72],
    "G Minor": [67, 70, 74],
    "A Minor": [69, 72, 76],
    "B Minor": [71, 74, 78],
    
    # 7th Chords
    "C7": [60, 64, 67, 70],
    "D7": [62, 66, 69, 72],
    "E7": [64, 68, 71, 74],
    "F7": [65, 69, 72, 75],
    "G7": [67, 71, 74, 77],
    "A7": [69, 73, 76, 79],
    "B7": [71, 75, 78, 81],
    
    # Major 7th Chords
    "Cmaj7": [60, 64, 67, 71],
    "Dmaj7": [62, 66, 69, 73],
    "Emaj7": [64, 68, 71, 75],
    "Fmaj7": [65, 69, 72, 76],
    "Gmaj7": [67, 71, 74, 78],
    "Amaj7": [69, 73, 76, 80],
    "Bmaj7": [71, 75, 78, 82],
    
    # Minor 7th Chords
    "Cm7": [60, 63, 67, 70],
    "Dm7": [62, 65, 69, 72],
    "Em7": [64, 67, 71, 74],
    "Fm7": [65, 68, 72, 75],
    "Gm7": [67, 70, 74, 77],
    "Am7": [69, 72, 76, 79],
    "Bm7": [71, 74, 78, 81],
    
    # Diminished Chords
    "Cdim": [60, 63, 66],
    "Ddim": [62, 65, 68],
    "Edim": [64, 67, 70],
    "Fdim": [65, 68, 71],
    "Gdim": [67, 70, 73],
    "Adim": [69, 72, 75],
    "Bdim": [71, 74, 77],
    
    # Augmented Chords
    "Caug": [60, 64, 68],
    "Daug": [62, 66, 70],
    "Eaug": [64, 68, 72],
    "Faug": [65, 69, 73],
    "Gaug": [67, 71, 75],
    "Aaug": [69, 73, 77],
    "Baug": [71, 75, 79],
    
    # Suspended Chords
    "Csus2": [60, 62, 67],
    "Dsus2": [62, 64, 69],
    "Esus2": [64, 66, 71],
    "Fsus2": [65, 67, 72],
    "Gsus2": [67, 69, 74],
    "Asus2": [69, 71, 76],
    "Bsus2": [71, 73, 78],
    
    "Csus4": [60, 65, 67],
    "Dsus4": [62, 67, 69],
    "Esus4": [64, 69, 71],
    "Fsus4": [65, 70, 72],
    "Gsus4": [67, 72, 74],
    "Asus4": [69, 74, 76],
    "Bsus4": [71, 76, 78],
    
    # 9th Chords
    "C9": [60, 64, 67, 70, 74],
    "D9": [62, 66, 69, 72, 76],
    "E9": [64, 68, 71, 74, 78],
    "F9": [65, 69, 72, 75, 79],
    "G9": [67, 71, 74, 77, 81],
    "A9": [69, 73, 76, 79, 83],
    "B9": [71, 75, 78, 81, 85],
}

# Chord Presets
CHORD_PRESETS = {
    "Pop": {  # I-V-vi-IV (C-G-Am-F)
        "left": {
            "thumb": {"name": "C", "chord": CHORD_LIBRARY["C Major"], "color": (255, 100, 100)},
            "index": {"name": "G", "chord": CHORD_LIBRARY["G Major"], "color": (255, 180, 100)},
            "middle": {"name": "Am", "chord": CHORD_LIBRARY["A Minor"], "color": (255, 255, 100)},
            "ring": {"name": "F", "chord": CHORD_LIBRARY["F Major"], "color": (100, 255, 100)},
            "pinky": {"name": "Dm", "chord": CHORD_LIBRARY["D Minor"], "color": (100, 200, 255)}
        },
        "right": {
            "thumb": {"name": "Em", "chord": CHORD_LIBRARY["E Minor"], "color": (255, 100, 180)},
            "index": {"name": "C7", "chord": CHORD_LIBRARY["C7"], "color": (255, 150, 200)},
            "middle": {"name": "G7", "chord": CHORD_LIBRARY["G7"], "color": (200, 100, 255)},
            "ring": {"name": "Am7", "chord": CHORD_LIBRARY["Am7"], "color": (150, 100, 255)},
            "pinky": {"name": "Fmaj7", "chord": CHORD_LIBRARY["Fmaj7"], "color": (180, 150, 255)}
        }
    },
    "Jazz": {  # ii-V-I jazz progression
        "left": {
            "thumb": {"name": "Dm7", "chord": CHORD_LIBRARY["Dm7"], "color": (255, 100, 100)},
            "index": {"name": "G7", "chord": CHORD_LIBRARY["G7"], "color": (255, 180, 100)},
            "middle": {"name": "Cmaj7", "chord": CHORD_LIBRARY["Cmaj7"], "color": (255, 255, 100)},
            "ring": {"name": "Am7", "chord": CHORD_LIBRARY["Am7"], "color": (100, 255, 100)},
            "pinky": {"name": "Em7", "chord": CHORD_LIBRARY["Em7"], "color": (100, 200, 255)}
        },
        "right": {
            "thumb": {"name": "Fmaj7", "chord": CHORD_LIBRARY["Fmaj7"], "color": (255, 100, 180)},
            "index": {"name": "Bm7", "chord": CHORD_LIBRARY["Bm7"], "color": (255, 150, 200)},
            "middle": {"name": "E7", "chord": CHORD_LIBRARY["E7"], "color": (200, 100, 255)},
            "ring": {"name": "A7", "chord": CHORD_LIBRARY["A7"], "color": (150, 100, 255)},
            "pinky": {"name": "D7", "chord": CHORD_LIBRARY["D7"], "color": (180, 150, 255)}
        }
    },
    "Blues": {  # 12-bar blues
        "left": {
            "thumb": {"name": "C7", "chord": CHORD_LIBRARY["C7"], "color": (255, 100, 100)},
            "index": {"name": "F7", "chord": CHORD_LIBRARY["F7"], "color": (255, 180, 100)},
            "middle": {"name": "G7", "chord": CHORD_LIBRARY["G7"], "color": (255, 255, 100)},
            "ring": {"name": "C9", "chord": CHORD_LIBRARY["C9"], "color": (100, 255, 100)},
            "pinky": {"name": "F9", "chord": CHORD_LIBRARY["F9"], "color": (100, 200, 255)}
        },
        "right": {
            "thumb": {"name": "G9", "chord": CHORD_LIBRARY["G9"], "color": (255, 100, 180)},
            "index": {"name": "Dm7", "chord": CHORD_LIBRARY["Dm7"], "color": (255, 150, 200)},
            "middle": {"name": "Em7", "chord": CHORD_LIBRARY["Em7"], "color": (200, 100, 255)},
            "ring": {"name": "Am7", "chord": CHORD_LIBRARY["Am7"], "color": (150, 100, 255)},
            "pinky": {"name": "Gm7", "chord": CHORD_LIBRARY["Gm7"], "color": (180, 150, 255)}
        }
    },
    "Rock": {  # Power chords and rock progressions
        "left": {
            "thumb": {"name": "C", "chord": CHORD_LIBRARY["C Major"], "color": (255, 100, 100)},
            "index": {"name": "F", "chord": CHORD_LIBRARY["F Major"], "color": (255, 180, 100)},
            "middle": {"name": "G", "chord": CHORD_LIBRARY["G Major"], "color": (255, 255, 100)},
            "ring": {"name": "Am", "chord": CHORD_LIBRARY["A Minor"], "color": (100, 255, 100)},
            "pinky": {"name": "Em", "chord": CHORD_LIBRARY["E Minor"], "color": (100, 200, 255)}
        },
        "right": {
            "thumb": {"name": "Dm", "chord": CHORD_LIBRARY["D Minor"], "color": (255, 100, 180)},
            "index": {"name": "Bb", "chord": CHORD_LIBRARY["B Major"], "color": (255, 150, 200)},
            "middle": {"name": "Csus4", "chord": CHORD_LIBRARY["Csus4"], "color": (200, 100, 255)},
            "ring": {"name": "Fsus2", "chord": CHORD_LIBRARY["Fsus2"], "color": (150, 100, 255)},
            "pinky": {"name": "Gsus4", "chord": CHORD_LIBRARY["Gsus4"], "color": (180, 150, 255)}
        }
    },
    "Classical": {  # Classical progressions
        "left": {
            "thumb": {"name": "C", "chord": CHORD_LIBRARY["C Major"], "color": (255, 100, 100)},
            "index": {"name": "Dm", "chord": CHORD_LIBRARY["D Minor"], "color": (255, 180, 100)},
            "middle": {"name": "Em", "chord": CHORD_LIBRARY["E Minor"], "color": (255, 255, 100)},
            "ring": {"name": "F", "chord": CHORD_LIBRARY["F Major"], "color": (100, 255, 100)},
            "pinky": {"name": "G", "chord": CHORD_LIBRARY["G Major"], "color": (100, 200, 255)}
        },
        "right": {
            "thumb": {"name": "Am", "chord": CHORD_LIBRARY["A Minor"], "color": (255, 100, 180)},
            "index": {"name": "Bdim", "chord": CHORD_LIBRARY["Bdim"], "color": (255, 150, 200)},
            "middle": {"name": "Cmaj7", "chord": CHORD_LIBRARY["Cmaj7"], "color": (200, 100, 255)},
            "ring": {"name": "Fmaj7", "chord": CHORD_LIBRARY["Fmaj7"], "color": (150, 100, 255)},
            "pinky": {"name": "G7", "chord": CHORD_LIBRARY["G7"], "color": (180, 150, 255)}
        }
    },
    "Gospel": {  # Gospel progressions
        "left": {
            "thumb": {"name": "Cmaj7", "chord": CHORD_LIBRARY["Cmaj7"], "color": (255, 100, 100)},
            "index": {"name": "Fmaj7", "chord": CHORD_LIBRARY["Fmaj7"], "color": (255, 180, 100)},
            "middle": {"name": "Gmaj7", "chord": CHORD_LIBRARY["Gmaj7"], "color": (255, 255, 100)},
            "ring": {"name": "Am7", "chord": CHORD_LIBRARY["Am7"], "color": (100, 255, 100)},
            "pinky": {"name": "Dm7", "chord": CHORD_LIBRARY["Dm7"], "color": (100, 200, 255)}
        },
        "right": {
            "thumb": {"name": "Em7", "chord": CHORD_LIBRARY["Em7"], "color": (255, 100, 180)},
            "index": {"name": "C9", "chord": CHORD_LIBRARY["C9"], "color": (255, 150, 200)},
            "middle": {"name": "F9", "chord": CHORD_LIBRARY["F9"], "color": (200, 100, 255)},
            "ring": {"name": "G9", "chord": CHORD_LIBRARY["G9"], "color": (150, 100, 255)},
            "pinky": {"name": "Bm7", "chord": CHORD_LIBRARY["Bm7"], "color": (180, 150, 255)}
        }
    },
    "Ambient": {  # Ambient/atmospheric
        "left": {
            "thumb": {"name": "Csus2", "chord": CHORD_LIBRARY["Csus2"], "color": (255, 100, 100)},
            "index": {"name": "Gsus2", "chord": CHORD_LIBRARY["Gsus2"], "color": (255, 180, 100)},
            "middle": {"name": "Fsus2", "chord": CHORD_LIBRARY["Fsus2"], "color": (255, 255, 100)},
            "ring": {"name": "Asus2", "chord": CHORD_LIBRARY["Asus2"], "color": (100, 255, 100)},
            "pinky": {"name": "Dsus2", "chord": CHORD_LIBRARY["Dsus2"], "color": (100, 200, 255)}
        },
        "right": {
            "thumb": {"name": "Esus2", "chord": CHORD_LIBRARY["Esus2"], "color": (255, 100, 180)},
            "index": {"name": "Cmaj7", "chord": CHORD_LIBRARY["Cmaj7"], "color": (255, 150, 200)},
            "middle": {"name": "Gmaj7", "chord": CHORD_LIBRARY["Gmaj7"], "color": (200, 100, 255)},
            "ring": {"name": "Fmaj7", "chord": CHORD_LIBRARY["Fmaj7"], "color": (150, 100, 255)},
            "pinky": {"name": "Amaj7", "chord": CHORD_LIBRARY["Amaj7"], "color": (180, 150, 255)}
        }
    },
    "Single Notes": {  # Original single note mode
        "left": {
            "thumb": {"name": "C4", "chord": [60], "color": (255, 100, 100)},
            "index": {"name": "D4", "chord": [62], "color": (255, 180, 100)},
            "middle": {"name": "E4", "chord": [64], "color": (255, 255, 100)},
            "ring": {"name": "F4", "chord": [65], "color": (100, 255, 100)},
            "pinky": {"name": "G4", "chord": [67], "color": (100, 200, 255)}
        },
        "right": {
            "thumb": {"name": "A4", "chord": [69], "color": (255, 100, 180)},
            "index": {"name": "B4", "chord": [71], "color": (255, 150, 200)},
            "middle": {"name": "C5", "chord": [72], "color": (200, 100, 255)},
            "ring": {"name": "D5", "chord": [74], "color": (150, 100, 255)},
            "pinky": {"name": "E5", "chord": [76], "color": (180, 150, 255)}
        }
    }
}

# Configuration
SUSTAIN_TIME = 0.5
NOTE_VELOCITY = 100
SHOW_LANDMARKS = True
current_preset = "Pop"
notes_config = CHORD_PRESETS[current_preset]
preset_names = list(CHORD_PRESETS.keys())
preset_index = 0

# Smaller circle sizes
ACTIVE_CIRCLE_RADIUS = 18  # Reduced from 28
ACTIVE_CIRCLE_BORDER = 22  # Reduced from 32
INACTIVE_CIRCLE_RADIUS = 10  # Reduced from 15

# State Management
prev_states = {hand: {finger: 0 for finger in ["thumb", "index", "middle", "ring", "pinky"]} for hand in ["left", "right"]}
active_notes = {}
note_history = deque(maxlen=15)
fps_history = deque(maxlen=30)
particle_effects = []

# Finger detection with thumb calibration
finger_state_buffer = {
    hand: {finger: deque(maxlen=3) for finger in ["thumb", "index", "middle", "ring", "pinky"]} 
    for hand in ["left", "right"]
}

class NoteParticle:
    def __init__(self, x, y, color, note_name):
        self.x = x
        self.y = y
        self.color = color
        self.note_name = note_name
        self.alpha = 255
        self.radius = 15  # Reduced from 20
        self.life = 30
        
    def update(self):
        self.y -= 3
        self.alpha -= 8
        self.radius += 0.8  # Slower growth
        self.life -= 1
        return self.life > 0

def finger_detection(hand, lm_list):
    # Finger detection with thumb handling
    fingers = []
    
    # Thumb detection
    if hand["type"] == "Right":
        # Right hand: thumb is open when tip (4) is to the LEFT of IP joint (3)
        if lm_list[4][0] < lm_list[3][0] - 10:  # Added threshold
            fingers.append(1)
        else:
            fingers.append(0)
    else:
        # Left hand: thumb is open when tip (4) is to the RIGHT of IP joint (3)
        if lm_list[4][0] > lm_list[3][0] + 10:  # Added threshold
            fingers.append(1)
        else:
            fingers.append(0)
    
    # Other four fingers
    tip_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]
    
    for tip, pip in zip(tip_ids, pip_ids):
        if lm_list[tip][1] < lm_list[pip][1] - 10:  # Added threshold for stability
            fingers.append(1)
        else:
            fingers.append(0)
    
    return fingers

def smooth_finger_state(hand_type, finger_name, current_state):
    # Smooth finger state to reduce jitter
    finger_state_buffer[hand_type][finger_name].append(current_state)
    
    # Use majority voting
    if len(finger_state_buffer[hand_type][finger_name]) >= 2:
        avg = sum(finger_state_buffer[hand_type][finger_name]) / len(finger_state_buffer[hand_type][finger_name])
        return 1 if avg > 0.5 else 0
    
    return current_state

def play_note(note_data, finger_name, hand_type, position):
    # Play a chord with visual feedback
    key = f"{hand_type}_{finger_name}"
    
    if key not in active_notes:
        chord = note_data["chord"]
        
        for note in chord:
            player.note_on(note, NOTE_VELOCITY)
        
        active_notes[key] = {
            "notes": chord,
            "time": time.time(),
            "color": note_data["color"],
            "name": note_data["name"]
        }
        
        note_history.append({
            "name": note_data["name"],
            "time": time.time(),
            "hand": hand_type
        })
        
        if position:
            particle_effects.append(NoteParticle(position[0], position[1], 
                                                 note_data["color"], note_data["name"]))

def stop_note_delayed(note_data, finger_name, hand_type):
    # Stop note after sustain time
    time.sleep(SUSTAIN_TIME)
    key = f"{hand_type}_{finger_name}"
    
    if key in active_notes:
        for note in note_data["chord"]:
            player.note_off(note, 0)
        del active_notes[key]

def draw_piano_keys(frame, h, w):
    # Draw visual piano keys at the bottom
    key_height = 100
    key_width = w // 10
    
    cv2.rectangle(frame, (0, h - key_height), (w, h), (20, 20, 20), -1)
    
    finger_names = ["thumb", "index", "middle", "ring", "pinky"]
    
    for i, (hand_type, hand_label) in enumerate([("right", "L"), ("left", "R")]):  # Swapped order
        for j, finger in enumerate(finger_names):
            x = (i * 5 + j) * key_width
            key = f"{hand_type}_{finger}"
            
            is_active = key in active_notes
            color = notes_config[hand_type][finger]["color"] if is_active else (50, 50, 50)
            
            cv2.rectangle(frame, (x + 3, h - key_height + 3), 
                         (x + key_width - 3, h - 3), color, -1)
            cv2.rectangle(frame, (x + 3, h - key_height + 3), 
                         (x + key_width - 3, h - 3), (200, 200, 200), 2)
            
            note_name = notes_config[hand_type][finger]["name"]
            text_size = cv2.getTextSize(note_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = x + (key_width - text_size[0]) // 2
            text_y = h - key_height + 35
            
            cv2.putText(frame, note_name, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            label_y = h - key_height + 65
            cv2.putText(frame, f"{hand_label}-{finger[0].upper()}", 
                       (x + 10, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
            
            # Show chord notes
            chord_text = f"{len(notes_config[hand_type][finger]['chord'])} notes"
            cv2.putText(frame, chord_text, (x + 5, h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

def draw_ui(frame, fps):
    # Draw UI overlay
    h, w = frame.shape[:2]
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (450, 280), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    cv2.putText(frame, "SEPIA - AIR PIANO", (20, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.3, (100, 200, 255), 3)
    
    cv2.putText(frame, f"Preset: {current_preset}", (20, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
    
    cv2.putText(frame, f"Active: {len(active_notes)}/10 chords", (20, 120),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 145),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    instructions = [
        "Controls:",
        "1-8 - Change Preset",
        "L - Toggle Landmarks",
        "R - Reset All Notes",
        "Q - Quit"
    ]
    
    y_offset = 170
    for instruction in instructions:
        cv2.putText(frame, instruction, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        y_offset += 22

def draw_note_history(frame, w):
    # Draw recent notes played
    if note_history:
        x_offset = w - 280
        y_offset = 20
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_offset - 10, 10), (w - 10, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        cv2.putText(frame, "Recent Chords:", (x_offset, y_offset + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        y_offset += 45
        for i, note in enumerate(list(note_history)[-7:]):
            age = time.time() - note["time"]
            alpha = max(50, 255 - int(age * 120))
            color = (alpha, alpha, alpha)
            
            text = f"{note['hand'][0].upper()}: {note['name']}"
            cv2.putText(frame, text, (x_offset, y_offset + i * 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

def draw_finger_indicators(frame, hands):
    # Draw visual indicators for finger states
    if not hands:
        return
    
    for hand in hands:
        hand_type = "right" if hand["type"] == "Left" else "left"
        lm_list = hand["lmList"]
        fingers = finger_detection(hand, lm_list)
        finger_names = ["thumb", "index", "middle", "ring", "pinky"]
        
        # Finger tip landmarks
        tip_ids = [4, 8, 12, 16, 20]
        
        for i, (finger_name, tip_id) in enumerate(zip(finger_names, tip_ids)):
            if tip_id < len(lm_list):
                x, y = lm_list[tip_id][0], lm_list[tip_id][1]
                
                # Get color based on finger state
                is_up = fingers[i] == 1
                note_data = notes_config[hand_type][finger_name]
                color = note_data["color"] if is_up else (100, 100, 100)
                
                # Draw circles
                if is_up:
                    cv2.circle(frame, (x, y), 25, color, -1)
                    cv2.circle(frame, (x, y), 28, (255, 255, 255), 3)
                    
                    # Draw note name
                    cv2.putText(frame, note_data["name"], (x - 15, y - 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                else:
                    cv2.circle(frame, (x, y), 15, color, 2)

def update_particles(frame):
    # Update and draw particle effects
    global particle_effects
    
    for particle in particle_effects[:]:
        if not particle.update():
            particle_effects.remove(particle)
        else:
            color = particle.color
            alpha = int(particle.alpha)
            overlay = frame.copy()
            cv2.circle(overlay, (int(particle.x), int(particle.y)), 
                      int(particle.radius), color, -1)
            cv2.addWeighted(overlay, alpha/255, frame, 1 - alpha/255, 0, frame)
            
            if particle.life > 20:
                cv2.putText(frame, particle.note_name, 
                           (int(particle.x) - 25, int(particle.y) - 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def change_preset(direction):
    # Change chord preset
    global preset_index, current_preset, notes_config, prev_states
    
    # Stop all active notes
    for key, note_info in list(active_notes.items()):
        for note in note_info["notes"]:
            player.note_off(note, 0)
    active_notes.clear()
    
    # Change preset
    preset_index = (preset_index + direction) % len(preset_names)
    current_preset = preset_names[preset_index]
    notes_config = CHORD_PRESETS[current_preset]
    
    # Reset states
    prev_states = {hand: {finger: 0 for finger in ["thumb", "index", "middle", "ring", "pinky"]} 
                   for hand in ["left", "right"]}
    
    print(f"Preset: {current_preset}")

print("=" * 70)
print("SEPIA - AIR PIANO - 10 Finger Chord System")
print("=" * 70)
print(f"Available Presets ({len(preset_names)}):")
for i, preset in enumerate(preset_names, 1):
    print(f"  {i}. {preset}")
print("=" * 70)
print("Show your hands to the camera and raise fingers to play chords!")
print("Press 1-8 to switch between presets")
print("Press Q to quit")
print("=" * 70)

try:
    while True:
        start_time = time.time()
        success, img = cap.read()
        if not success:
            continue
            
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        # Detect hands
        hands, img = detector.findHands(img, draw=SHOW_LANDMARKS, flipType=False)
        
        # Process hands
        if hands:
            for hand in hands:
                hand_type = "right" if hand["type"] == "Left" else "left"
                fingers = finger_detection(hand, hand["lmList"])
                lm_list = hand["lmList"]
                
                finger_names = ["thumb", "index", "middle", "ring", "pinky"]
                tip_ids = [4, 8, 12, 16, 20]
                
                for i, finger_name in enumerate(finger_names):
                    is_up = fingers[i]
                    prev_state = prev_states[hand_type][finger_name]
                    
                    # Finger raised - play chord
                    if is_up == 1 and prev_state == 0:
                        note_data = notes_config[hand_type][finger_name]
                        position = None
                        
                        if tip_ids[i] < len(lm_list):
                            position = (lm_list[tip_ids[i]][0], lm_list[tip_ids[i]][1])
                        
                        play_note(note_data, finger_name, hand_type, position)
                        
                        # Start thread to stop note after sustain time
                        threading.Thread(
                            target=stop_note_delayed,
                            args=(note_data, finger_name, hand_type),
                            daemon=True
                        ).start()
                    
                    prev_states[hand_type][finger_name] = is_up
        
        # Draw UI elements
        draw_piano_keys(img, h, w)
        draw_finger_indicators(img, hands)
        update_particles(img)
        
        # Calculate FPS
        fps = 1 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
        fps_history.append(fps)
        avg_fps = sum(fps_history) / len(fps_history)
        
        draw_ui(img, avg_fps)
        draw_note_history(img, w)
        
        # Display
        cv2.imshow("Sepia - Air Piano", img)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('l'):
            SHOW_LANDMARKS = not SHOW_LANDMARKS
            print(f"Landmarks: {'ON' if SHOW_LANDMARKS else 'OFF'}")
        elif key == ord('r'):
            # Reset all notes
            for key_name, note_info in list(active_notes.items()):
                for note in note_info["notes"]:
                    player.note_off(note, 0)
            active_notes.clear()
            print("All notes reset!")
        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8')]:
            # Switch preset
            preset_num = int(chr(key))
            if preset_num <= len(preset_names):
                preset_index = preset_num - 1
                current_preset = preset_names[preset_index]
                notes_config = CHORD_PRESETS[current_preset]
                
                # Stop all active notes
                for key_name, note_info in list(active_notes.items()):
                    for note in note_info["notes"]:
                        player.note_off(note, 0)
                active_notes.clear()
                
                # Reset states
                prev_states = {hand: {finger: 0 for finger in ["thumb", "index", "middle", "ring", "pinky"]} 
                             for hand in ["left", "right"]}
                
                print(f"Switched to: {current_preset}")

except KeyboardInterrupt:
    print("\nShutting down gracefully...")

finally:
    # Cleanup
    for key, note_info in list(active_notes.items()):
        for note in note_info["notes"]:
            player.note_off(note, 0)
    player.close()
    pygame.midi.quit()
    cap.release()
    cv2.destroyAllWindows()
    print("Goodbye!")
