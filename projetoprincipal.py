import cv2
import mediapipe as mp
import pyautogui
import math
import time
import os
import subprocess

# ================== CONFIG ==================
pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

SMOOTHING = 0.25
MOUSE_SENS = 2.0
PINCH_DIST = 40
PINCH_MOVE_FACTOR = 0.25

MENU_RADIUS = 90
MENU_COOLDOWN = 0.6
# ============================================

menu_items = [
    {"name":"PRINT", "angle":270},
    {"name":"WHATS", "angle":210},
    {"name":"DISCORD", "angle":330}
]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

mouse_x, mouse_y = screen_w // 2, screen_h // 2
clicking = False

menu_open = False
menu_center = (0, 0)
last_menu_time = 0
last_choice = None

# ================== FUNÇÕES ==================
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def is_pinch(lm, w, h):
    thumb = (lm[4].x * w, lm[4].y * h)
    index = (lm[8].x * w, lm[8].y * h)
    return dist(thumb, index) < PINCH_DIST

def draw_menu(frame, center, hover):
    for item in menu_items:
        ang = math.radians(item["angle"])
        x = int(center[0] + MENU_RADIUS * math.cos(ang))
        y = int(center[1] + MENU_RADIUS * math.sin(ang))
        color = (0,255,0) if item["name"] == hover else (255,255,255)
        cv2.circle(frame, (x, y), 30, color, 2)
        cv2.putText(frame, item["name"], (x-30, y+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

def detect_menu(finger, center):
    for item in menu_items:
        ang = math.radians(item["angle"])
        x = int(center[0] + MENU_RADIUS * math.cos(ang))
        y = int(center[1] + MENU_RADIUS * math.sin(ang))
        if dist(finger, (x, y)) < 30:
            return item["name"]
    return None

def execute(name):
    if name == "PRINT":
        pyautogui.screenshot(f"print_{int(time.time())}.png")

    elif name == "WHATS":
        os.startfile("whatsapp:")

    elif name == "DISCORD":
        base = os.path.expandvars(r"%LOCALAPPDATA%\Discord")
        if os.path.exists(base):
            for f in os.listdir(base):
                if f.startswith("app-"):
                    exe = os.path.join(base, f, "Discord.exe")
                    if os.path.exists(exe):
                        subprocess.Popen(exe)
                        return

# ================== LOOP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    right_index = None
    left_index = None
    right_pinch = False
    left_pinch = False

    if res.multi_hand_landmarks and res.multi_handedness:
        for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
            label = handed.classification[0].label
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            ix = int(lm.landmark[8].x * w)
            iy = int(lm.landmark[8].y * h)

            # ===== MÃO DIREITA → MOUSE =====
            if label == "Right":
                right_index = (ix, iy)
                right_pinch = is_pinch(lm.landmark, w, h)

                factor = PINCH_MOVE_FACTOR if right_pinch else 1.0

                tx = ix * screen_w / w * MOUSE_SENS
                ty = iy * screen_h / h * MOUSE_SENS

                mouse_x += (tx - mouse_x) * SMOOTHING * factor
                mouse_y += (ty - mouse_y) * SMOOTHING * factor

                mouse_x = max(0, min(screen_w - 1, mouse_x))
                mouse_y = max(0, min(screen_h - 1, mouse_y))

                pyautogui.moveTo(int(mouse_x), int(mouse_y), _pause=False)

                # Clique com pinça
                if right_pinch and not clicking:
                    pyautogui.mouseDown()
                    clicking = True
                    cv2.circle(frame, (ix, iy), 20, (0, 0, 255), -1)

                elif not right_pinch and clicking:
                    pyautogui.mouseUp()
                    clicking = False

            # ===== MÃO ESQUERDA → MENU =====
            if label == "Left":
                left_index = (ix, iy)
                left_pinch = is_pinch(lm.landmark, w, h)

    # ===== MENU LOGIC =====
    now = time.time()
    if left_index:
        if left_pinch and not menu_open and now - last_menu_time > MENU_COOLDOWN:
            menu_open = True
            menu_center = left_index

        if menu_open:
            last_choice = detect_menu(left_index, menu_center)
            draw_menu(frame, menu_center, last_choice)

            if not left_pinch:
                if last_choice:
                    execute(last_choice)
                menu_open = False
                last_menu_time = now
                last_choice = None

    cv2.imshow("Hand Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
        break

cap.release()
cv2.destroyAllWindows()