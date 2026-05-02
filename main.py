"""
AI Traffic Guardian — 5 second delay before car enters, speed 50.
Press Q to quit.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import random
import time

BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "pose_landmarker_heavy.task")
CAR_PATH   = os.path.join(BASE_DIR, "car.webp")

if not os.path.exists(MODEL_PATH):
    print("ERROR: Model file not found!")
    exit(1)
if not os.path.exists(CAR_PATH):
    print("ERROR: car.webp not found!")
    exit(1)

print("Loading car image...")
car_img_raw = cv2.imread(CAR_PATH, cv2.IMREAD_UNCHANGED)
if car_img_raw is None:
    print("ERROR: Could not read car.webp")
    exit(1)

def remove_white_background(img):
    if img.shape[2] == 4:
        return img
    bgra     = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask  = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    mask_inv = cv2.GaussianBlur(mask_inv, (3, 3), 0)
    bgra[:, :, 3] = mask_inv
    return bgra

car_img     = remove_white_background(car_img_raw)
CAR_WIDTH   = 260
car_h_orig, car_w_orig = car_img.shape[:2]
CAR_H       = int(car_h_orig * (CAR_WIDTH / car_w_orig))
car_resized = cv2.resize(car_img, (CAR_WIDTH, CAR_H), interpolation=cv2.INTER_AREA)

# ── Timing constants ──────────────────────────────────────────
CAR_SPEED        = 50
CAR_WAIT_SECONDS = 5.0   # ✅ wait 5 seconds before car enters
PARTICLE_SECONDS = 4.5
SAFETY_SECONDS   = 90.0

SAFETY_MESSAGES = [
    ("STOP.",     "Look before you cross the road.",               "Every second counts. Stay safe."),
    ("DANGER!",   "Never use your phone while crossing.",          "Distraction kills. Stay alert."),
    ("WARNING!",  "Always use the zebra crossing.",                "Your life is precious. Cross safely."),
    ("ALERT!",    "Speed kills. Pedestrians have no protection.",  "Slow down. Save lives."),
    ("CAUTION!",  "Never run across the road.",                    "Take your time. Reach safely."),
    ("REMEMBER!", "Make eye contact with drivers before crossing.","Be visible. Be safe."),
]

def overlay_image(background, overlay, x, y):
    oh, ow = overlay.shape[:2]
    bh, bw = background.shape[:2]
    x1 = max(x, 0);  y1 = max(y, 0)
    x2 = min(x + ow, bw); y2 = min(y + oh, bh)
    if x2 <= x1 or y2 <= y1:
        return background
    ox1 = x1 - x; oy1 = y1 - y
    ox2 = ox1 + (x2 - x1); oy2 = oy1 + (y2 - y1)
    roi      = background[y1:y2, x1:x2].astype(np.float32)
    car_crop = overlay[oy1:oy2, ox1:ox2]
    alpha    = car_crop[:, :, 3:4].astype(np.float32) / 255.0
    rgb      = car_crop[:, :, :3].astype(np.float32)
    background[y1:y2, x1:x2] = (rgb * alpha + roi * (1.0 - alpha)).astype(np.uint8)
    return background

def draw_centered_text(frame, text, y, font_scale, color, thickness, outline=(0,0,0)):
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    tsz  = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x    = (w - tsz[0]) // 2
    cv2.putText(frame, text, (x-2, y), font, font_scale, outline, thickness+3)
    cv2.putText(frame, text, (x,   y), font, font_scale, color,   thickness)

def draw_safety_screen(frame, message_idx, elapsed, total):
    h, w  = frame.shape[:2]
    msg   = SAFETY_MESSAGES[message_idx % len(SAFETY_MESSAGES)]
    pulse = int(180 + 60 * abs(np.sin(elapsed * 5)))
    frame[:] = (0, 0, pulse)
    cv2.rectangle(frame, (0, 0),      (w, 80),   (0, 0, 80), -1)
    cv2.rectangle(frame, (0, h - 80), (w, h),    (0, 0, 80), -1)
    draw_centered_text(frame, "AI TRAFFIC GUARDIAN", 50, 0.9, (255,255,255), 2)
    tri_pts = np.array([
        [w//2,      h//4 - 40],
        [w//2 - 55, h//4 + 35],
        [w//2 + 55, h//4 + 35],
    ], np.int32)
    cv2.fillPoly(frame, [tri_pts], (255, 255, 0))
    cv2.polylines(frame, [tri_pts], True, (255, 200, 0), 3)
    draw_centered_text(frame, "!", h//4 + 22, 1.6, (0,0,0), 3)
    draw_centered_text(frame, msg[0], h//4 + 95,  2.2,  (255,255,255), 3)
    cv2.line(frame, (w//6, h//4+115), (5*w//6, h//4+115), (255,255,255), 2)
    draw_centered_text(frame, msg[1], h//2 + 35,  0.9,  (255,255,200), 2)
    draw_centered_text(frame, msg[2], h//2 + 85,  0.85, (255,220,180), 2)
    secs_left = max(0, int(total - elapsed))
    draw_centered_text(frame, f"Resuming in {secs_left}s", h-55, 0.65, (255,200,200), 1)
    if int(elapsed * 2) % 2 == 0:
        draw_centered_text(frame, "COLLISION DETECTED — STAY SAFE ON ROADS",
                           h-25, 0.6, (255,255,0), 1)
    return frame

def draw_skeleton(black, landmarks_list, w, h):
    for landmarks in landmarks_list:
        for s, e in CONNECTIONS:
            if s < len(landmarks) and e < len(landmarks):
                x1 = int(landmarks[s].x * w)
                y1 = int(landmarks[s].y * h)
                x2 = int(landmarks[e].x * w)
                y2 = int(landmarks[e].y * h)
                cv2.line(black, (x1,y1), (x2,y2), (0,200,255), 2)
        for lm in landmarks:
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            cv2.circle(black, (cx,cy), 5, (0,255,0), -1)

print("Loading pose model...")
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options      = vision.PoseLandmarkerOptions(
    base_options=base_options,
    num_poses=10,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_segmentation_masks=False
)
detector = vision.PoseLandmarker.create_from_options(options)
print("Model loaded. Starting webcam...\n")

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(24,26),
    (25,27),(26,28),(27,29),(28,30),(29,31),(30,32)
]

STATE_IDLE    = "idle"
STATE_WAITING = "waiting"   # ✅ new — waiting 5 seconds before car enters
STATE_DRIVING = "driving"
STATE_FLYING  = "flying"
STATE_SAFETY  = "safety"

state          = STATE_IDLE
car_x          = 0
impact_x       = 0
particles      = []
last_landmarks = None
crash_cooldown = 0
collision_time = None
safety_start   = None
wait_start     = None   # ✅ when pedestrian was first detected
message_idx    = 0

def create_particles(landmarks, w, h):
    pts = []
    for lm in landmarks:
        pts.append({
            "x":  float(int(lm.x * w)),
            "y":  float(int(lm.y * h)),
            "vx": random.uniform(5, 18),
            "vy": random.uniform(-16, -3),
            "color": (random.randint(0,255), random.randint(100,255), random.randint(0,255)),
            "size": random.randint(4, 11)
        })
    return pts

def update_particles(pts):
    for p in pts:
        p["x"]  += p["vx"]
        p["y"]  += p["vy"]
        p["vy"] += 0.65
        p["vx"] *= 0.97

def draw_hud(frame, count, cur_state, wait_left=0):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (w,50), (0,0,0), -1)
    cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, "AI TRAFFIC GUARDIAN", (10,33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2)
    label = f"Pedestrians: {count}"
    tsz   = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.putText(frame, label, (w-tsz[0]-10, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,100), 2)
    if cur_state == STATE_FLYING:
        cv2.putText(frame, "!! COLLISION DETECTED !!",
                    (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    elif cur_state == STATE_WAITING:
        # ✅ Show countdown while waiting
        secs = max(0, int(wait_left))
        cv2.putText(frame, f"WARNING: Car approaching in {secs}s !",
                    (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
    elif count > 0:
        cv2.putText(frame, f"! {count} PEDESTRIAN(S) DETECTED !",
                    (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,80,255), 2)
    cv2.putText(frame, "Press Q to quit", (10, h-40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
    return frame

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit(1)

print("Webcam started! Press Q to quit.")
print(f"Car speed: {CAR_SPEED} | Car wait: {CAR_WAIT_SECONDS}s")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    now  = time.time()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results   = detector.detect(mp_image)

    pedestrian_count = len(results.pose_landmarks) if results.pose_landmarks else 0
    if results.pose_landmarks:
        last_landmarks = results.pose_landmarks[0]

    black        = np.zeros((h, w, 3), dtype=np.uint8)
    car_ground_y = int(h * 0.75)
    car_top_y    = car_ground_y - CAR_H

    if crash_cooldown > 0:
        crash_cooldown -= 1

    # ── SAFETY SCREEN ─────────────────────────────────────────
    if state == STATE_SAFETY:
        elapsed = now - safety_start
        draw_safety_screen(black, message_idx, elapsed, SAFETY_SECONDS)
        if elapsed >= SAFETY_SECONDS:
            state          = STATE_IDLE
            crash_cooldown = 60
        cv2.imshow("AI Traffic Guardian", black)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # ── FLYING PARTICLES ──────────────────────────────────────
    if state == STATE_FLYING:
        elapsed = now - collision_time
        if elapsed >= PARTICLE_SECONDS:
            state        = STATE_SAFETY
            safety_start = time.time()
        else:
            overlay_image(black, car_resized, int(impact_x), car_top_y)
            update_particles(particles)
            for p in particles:
                if p["y"] < h + 30:
                    cv2.circle(black, (int(p["x"]), int(p["y"])),
                               p["size"], p["color"], -1)
                    tx = int(p["x"] - p["vx"] * 2)
                    ty = int(p["y"] - p["vy"] * 2)
                    cv2.line(black, (int(p["x"]), int(p["y"])),
                             (tx, ty), p["color"], 1)
            draw_hud(black, pedestrian_count, state)
            cv2.imshow("AI Traffic Guardian", black)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

    # ── IDLE ──────────────────────────────────────────────────
    if state == STATE_IDLE:
        if pedestrian_count > 0 and crash_cooldown <= 0:
            # ✅ Start waiting instead of driving immediately
            state       = STATE_WAITING
            wait_start  = time.time()
            message_idx = random.randint(0, len(SAFETY_MESSAGES)-1)
        if results.pose_landmarks:
            draw_skeleton(black, results.pose_landmarks, w, h)

    # ── WAITING — show skeleton + countdown ───────────────────
    if state == STATE_WAITING:
        elapsed   = now - wait_start
        wait_left = CAR_WAIT_SECONDS - elapsed
        if results.pose_landmarks:
            draw_skeleton(black, results.pose_landmarks, w, h)
        if elapsed >= CAR_WAIT_SECONDS:
            # ✅ 5 seconds done — now car enters
            state = STATE_DRIVING
            car_x = -CAR_WIDTH - 10
        else:
            draw_hud(black, pedestrian_count, STATE_WAITING, wait_left)
            cv2.imshow("AI Traffic Guardian", black)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

    # ── DRIVING ───────────────────────────────────────────────
    if state == STATE_DRIVING:
        car_x += CAR_SPEED
        skeleton_x = w // 2
        if last_landmarks and len(last_landmarks) > 23:
            skeleton_x = int(last_landmarks[23].x * w) - 60
        if car_x + CAR_WIDTH >= skeleton_x:
            state          = STATE_FLYING
            impact_x       = car_x
            collision_time = time.time()
            if last_landmarks:
                particles = create_particles(last_landmarks, w, h)
        else:
            if results.pose_landmarks:
                draw_skeleton(black, results.pose_landmarks, w, h)
            overlay_image(black, car_resized, int(car_x), car_top_y)

    draw_hud(black, pedestrian_count, state)
    cv2.imshow("AI Traffic Guardian", black)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")
