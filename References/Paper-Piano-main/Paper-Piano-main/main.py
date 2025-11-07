import cv2
import mediapipe as mp
import pygame
import time
import math
import numpy as np

# =============================================================================
# Variables Globales y Configuración Inicial
# =============================================================================
selected_points = []     # 4 puntos para transformación en perspectiva
static_keys = False      # Modo: teclas fijas (estáticas)
frontal_view = False     # Vista frontal (transformada)
rotate_180 = False       # Rota el frame 180°
mirror_effect = False    # Efecto espejo

# Configuración de octavas y notas
current_octave = 4
min_octave = 1
max_octave = 7
num_octavas = 1          # 1 (12 teclas) o 2 (24 teclas)
notes_in_octave = ["C", "Db", "D", "Eb", "E", "F",
                   "Gb", "G", "Ab", "A", "Bb", "B"]
octave_sounds = []

# Diccionario de botones de UI
button_height = 30
button_font_scale = 0.5
button_thickness = 1
buttons = {
    "reset":    {"pos": (10, 10),  "size": (60, button_height),
                 "bg": (50, 50, 50), "text": "Rst"},
    "toggle":   {"pos": (80, 10),  "size": (70, button_height),
                 "bg": (50, 50, 50), "text": "Togl"},
    "frontal":  {"pos": (160, 10), "size": (70, button_height),
                 "bg": (50, 50, 50), "text": "Frnt"},
    "oct_down": {"pos": (240, 10), "size": (60, button_height),
                 "bg": (50, 50, 50), "text": "Oct-"},
    "oct_up":   {"pos": (310, 10), "size": (60, button_height),
                 "bg": (50, 50, 50), "text": "Oct+"},
}
text_color = (255, 255, 255)

# Teclas dinámicas o estáticas
notes_detected = {}
last_triggered = {}

# Controles extra (los dos discos verdes de Oct-/Oct+)
control_positions = {"down": None, "up": None}
control_radii     = {"down": 0,      "up": 0}
last_control      = {"down": 0.0,    "up": 0.0}
CONTROL_COOLDOWN  = 0.5
NOTE_COOLDOWN     = 0.5

# Para “snapshot” en modo estático
static_control_positions = {"down": None, "up": None}
static_control_radii     = {"down": 0,      "up": 0}
static_note_blobs        = []    # lista de (pos, radio)
prev_static_keys         = False

# =============================================================================
# Funciones Auxiliares
# =============================================================================
def draw_button(frame, button_key, active_text=None):
    btn = buttons[button_key]
    x, y = btn["pos"]; w, h = btn["size"]
    text = active_text or btn["text"]
    cv2.rectangle(frame, (x, y), (x + w, y + h), btn["bg"], -1)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                  button_font_scale, button_thickness)
    tx = x + (w - tw)//2; ty = y + (h + th)//2
    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                button_font_scale, text_color, button_thickness, cv2.LINE_AA)

def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
    return rect

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# =============================================================================
# Callback de Mouse
# =============================================================================
def click_event(event, x, y, flags, param):
    global selected_points, static_keys, frontal_view
    global current_octave, num_octavas, prev_static_keys
    if event != cv2.EVENT_LBUTTONDOWN: return

    for k, btn in buttons.items():
        bx, by = btn["pos"]; bw, bh = btn["size"]
        if bx <= x <= bx + bw and by <= y <= by + bh:
            if k == "reset":
                selected_points.clear()
                print("Reset puntos.")
            elif k == "toggle":
                static_keys = not static_keys
                print("Modo estático." if static_keys else "Modo dinámico.")
                if not static_keys:
                    static_note_blobs.clear()
                    static_control_positions["down"] = static_control_positions["up"] = None
                    static_control_radii["down"] = static_control_radii["up"] = 0
            elif k == "frontal":
                frontal_view = not frontal_view
                print("Vista frontal." if frontal_view else "Normal.")
            elif k == "oct_down":
                if current_octave > min_octave:
                    current_octave -= 1; update_octave_sounds()
                    print("Oct bajada:", current_octave)
                else:
                    print("Oct mínima.")
            elif k == "oct_up":
                if ((num_octavas == 1 and current_octave < max_octave) or
                    (num_octavas == 2 and current_octave < max_octave - 1)):
                    current_octave += 1; update_octave_sounds()
                    print("Oct subida:", current_octave)
                else:
                    print("Oct máxima.")
            prev_static_keys = static_keys
            return

    if len(selected_points) < 4:
        selected_points.append((x, y))
        print("Punto:", (x, y))

# =============================================================================
# Carga y mapeo de sonidos
# =============================================================================
pygame.mixer.init()
def load_sound(path):
    try: return pygame.mixer.Sound(path)
    except Exception as e:
        print(f"Error cargando {path}: {e}")
        return None

def update_octave_sounds():
    global octave_sounds, notes_detected, last_triggered
    temp = []
    octs = [current_octave] if num_octavas == 1 else [current_octave, current_octave+1]
    for o in octs:
        for n in notes_in_octave:
            fn = f"Piano_Samples/{n}{o}.mp3"
            s = load_sound(fn)
            if s is None:
                print("Falta muestra:", fn, "–> cancelado")
                return
            temp.append(s)
    octave_sounds[:] = temp
    print("Sonidos para octavas", octs)
    if static_keys and notes_detected:
        items = sorted(notes_detected.items(), key=lambda it: it[1]["pos"][0])
        needed = 12 * num_octavas
        if len(items) == needed:
            newd = {}
            for i, (old, d) in enumerate(items):
                oi = (current_octave if num_octavas==1
                      else current_octave + i//12)
                nl = notes_in_octave[i % 12]
                nm = f"{nl}{oi}"
                d["sound"] = octave_sounds[i]
                newd[nm] = d
                last_triggered[nm] = last_triggered.pop(old, 0)
            notes_detected.clear(); notes_detected.update(newd)
        else:
            print("Remapeo estático falló, limpiando notas.")
            notes_detected.clear(); last_triggered.clear()

update_octave_sounds()

# =============================================================================
# Mediapipe y OpenCV Init
# =============================================================================
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(static_image_mode=False, max_num_hands=3,
                          min_detection_confidence=0.7,
                          min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No hay cámara."); exit()
cv2.namedWindow("Paper Piano", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Paper Piano", click_event)

# Trackbars HSV
def nothing(x): pass
cv2.namedWindow("HSV Adjust", cv2.WINDOW_AUTOSIZE)
for param, init, m in [("Hmin",50,179),("Smin",100,255),("Vmin",20,255),
                       ("Hmax",100,179),("Smax",255,255),("Vmax",150,255)]:
    cv2.createTrackbar(param, "HSV Adjust", init, m, nothing)

HSV_presets = {
    "Verde Oscuro":{"Hmin":85,"Smin":132,"Vmin":113,"Hmax":99,"Smax":255,"Vmax":255},
    "Verde Medio": {"Hmin":35,"Smin":80, "Vmin":50, "Hmax":85,"Smax":255,"Vmax":255},
    "Verde Claro": {"Hmin":50,"Smin":50, "Vmin":80, "Hmax":71,"Smax":82, "Vmax":255},
}
def set_HSV_trackbars(p):
    for k,v in p.items():
        cv2.setTrackbarPos(k, "HSV Adjust", v)
    print("Preset HSV:", p)

# =============================================================================
# Bucle Principal
# =============================================================================
while True:
    ret, frame = cap.read()
    if not ret: break

    # 1) Dibujar UI
    for k in buttons: draw_button(frame, k)
    info_oct = (f"Oct:{current_octave}" if num_octavas==1
                else f"Oct:{current_octave} y {current_octave+1}")
