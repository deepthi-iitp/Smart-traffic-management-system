import cv2, time
import numpy as np
from ultralytics import YOLO
import lgpio

# ================================================================== #
#  VEHICLE CLASSES                                                     #
# ================================================================== #
VEHICLE_CLASSES = {
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    67: "toy_car",   # top-down toy car detected as cell phone
    9:  "toy_car",   # traffic light misdetect
}

# ================================================================== #
#  AMBULANCE DETECTION — WHITE COLOUR ONLY                            #
#                                                                      #
#  Your ambulance is WHITE with red stripe text.                      #
#  We detect it by checking if the bounding box crop has a large      #
#  proportion of WHITE pixels in HSV space.                           #
#                                                                      #
#  Other vehicles:                                                     #
#    Blue  (Police)      → NOT emergency                              #
#    Orange (School Bus) → NOT emergency                              #
#    Yellow (DHL van)    → NOT emergency                              #
#    Green (Fruit truck) → NOT emergency                              #
#    White (Ambulance)   → EMERGENCY ✓                                #
# ================================================================== #

# White in HSV: low saturation, high value
WHITE_S_MAX  = 60    # saturation must be LOW  (white has near-zero saturation)
WHITE_V_MIN  = 180   # value must be HIGH       (white is bright)
WHITE_RATIO  = 0.30  # at least 30% of box pixels must be white

# Minimum YOLO confidence to run colour check
EMERGENCY_CONF = 0.10

# How long to keep emergency GREEN after ambulance leaves frame
EMERGENCY_HOLD_TIME = 8   # seconds

# ================================================================== #
#  LANE POLYGONS  (320×240 display space)                             #
# ================================================================== #
LANE_POLYGONS = {
    "lane2": np.array([[0,0],[155,0],[155,240],[0,240]], dtype=np.int32),
    "lane3": np.array([[160,0],[320,0],[320,240],[160,240]], dtype=np.int32),
}

CALIBRATE = False

# ================================================================== #
#  MODEL                                                               #
# ================================================================== #
model = YOLO("yolov8n.pt")

MIN_GREEN   = 3
MAX_GREEN   = 7
YELLOW_TIME = 1

# ================================================================== #
#  GPIO                                                                #
# ================================================================== #
CHIP = lgpio.gpiochip_open(0)

LED_PINS = {
    "lane2": {"R": 5,  "Y": 6,  "G": 13},
    "lane3": {"R": 19, "Y": 26, "G": 21},
}

for lane in LED_PINS:
    for color in LED_PINS[lane]:
        pin = LED_PINS[lane][color]
        try:
            lgpio.gpio_claim_output(CHIP, pin, 0)
        except lgpio.error:
            lgpio.gpio_free(CHIP, pin)
            lgpio.gpio_claim_output(CHIP, pin, 0)
        print(f"[GPIO] Pin {pin} ({lane} {color}) claimed OK")

# ================================================================== #
#  CAMERA                                                              #
# ================================================================== #
cap = cv2.VideoCapture("http://192.173.4.129:8080/video")

# ================================================================== #
#  STATE                                                               #
# ================================================================== #
state = {
    "active_lane": "lane2",
    "signal":      "GREEN",
    "timer_start": time.time(),
    "green_time":  MIN_GREEN,
    "counts":      {"lane2": 0, "lane3": 0},
    "cycle_order": ["lane2", "lane3"],
    "cycle_index": 0,
}

# Emergency state tracker
emergency = {
    "active":     False,   # True while ambulance is visible in frame
    "lane":       None,    # which lane ambulance is in
    "hold_until": 0,       # keep GREEN until this time even after ambulance leaves
}

counted_ids    = {"lane2": set(), "lane3": set()}
last_reset     = time.time()
frame_count    = 0
id_frame_count = {}
STABLE_FRAMES  = 2

# Flash for emergency LEDs
flash_state     = True
last_flash_time = time.time()
FLASH_INTERVAL  = 0.4

calib_points = []
calib_lane   = "lane2"

# ================================================================== #
#  WHITE COLOUR DETECTION                                              #
# ================================================================== #
def is_white_vehicle(crop_bgr):
    """
    Returns True if the bounding box crop is predominantly white.
    Works by checking HSV: low saturation + high brightness = white.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return False

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    # White pixels: saturation < WHITE_S_MAX AND value > WHITE_V_MIN
    mask = cv2.inRange(
        hsv,
        np.array([0,   0,         WHITE_V_MIN], dtype=np.uint8),
        np.array([180, WHITE_S_MAX, 255],        dtype=np.uint8)
    )
    ratio = np.count_nonzero(mask) / max(mask.size, 1)
    return ratio >= WHITE_RATIO

# ================================================================== #
#  LED CONTROL                                                         #
# ================================================================== #
def update_leds():
    global flash_state, last_flash_time

    now = time.time()
    if now - last_flash_time >= FLASH_INTERVAL:
        flash_state     = not flash_state
        last_flash_time = now

    in_emergency = emergency["active"] or time.time() < emergency["hold_until"]

    for lane in LED_PINS:
        pins = LED_PINS[lane]

        if in_emergency:
            # Flash green on priority lane, flash red on other lane
            if lane == state["active_lane"]:
                lgpio.gpio_write(CHIP, pins["G"], 1 if flash_state else 0)
                lgpio.gpio_write(CHIP, pins["Y"], 0)
                lgpio.gpio_write(CHIP, pins["R"], 0)
            else:
                lgpio.gpio_write(CHIP, pins["R"], 1 if flash_state else 0)
                lgpio.gpio_write(CHIP, pins["G"], 0)
                lgpio.gpio_write(CHIP, pins["Y"], 0)
        else:
            # Normal signal
            if lane == state["active_lane"]:
                if state["signal"] == "GREEN":
                    lgpio.gpio_write(CHIP, pins["G"], 1)
                    lgpio.gpio_write(CHIP, pins["Y"], 0)
                    lgpio.gpio_write(CHIP, pins["R"], 0)
                elif state["signal"] == "YELLOW":
                    lgpio.gpio_write(CHIP, pins["Y"], 1)
                    lgpio.gpio_write(CHIP, pins["G"], 0)
                    lgpio.gpio_write(CHIP, pins["R"], 0)
            else:
                lgpio.gpio_write(CHIP, pins["R"], 1)
                lgpio.gpio_write(CHIP, pins["G"], 0)
                lgpio.gpio_write(CHIP, pins["Y"], 0)

# ================================================================== #
#  SIGNAL TIMING                                                       #
# ================================================================== #
def calc_green_time(count):
    if count == 0:  return 3
    if count < 3:   return 4
    if count < 6:   return 6
    return MAX_GREEN   # 7s for heavy traffic

def next_lane():
    state["cycle_index"] = (state["cycle_index"] + 1) % 2
    next_l = state["cycle_order"][state["cycle_index"]]
    gt = calc_green_time(state["counts"][next_l])
    state["active_lane"] = next_l
    state["signal"]      = "GREEN"
    state["green_time"]  = gt
    state["timer_start"] = time.time()
    print(f"[SIGNAL] Switching to {next_l.upper()} for {gt}s")

def check_signal_timing():
    # Freeze normal timing during emergency or hold period
    if emergency["active"] or time.time() < emergency["hold_until"]:
        return

    elapsed = time.time() - state["timer_start"]
    if state["signal"] == "GREEN" and elapsed >= state["green_time"]:
        state["signal"]      = "YELLOW"
        state["timer_start"] = time.time()
        print(f"[SIGNAL] {state['active_lane'].upper()} -> YELLOW")
    elif state["signal"] == "YELLOW" and elapsed >= YELLOW_TIME:
        next_lane()

def trigger_emergency(lane):
    """Immediately give GREEN to ambulance lane."""
    was_active = emergency["active"]
    emergency["active"]     = True
    emergency["lane"]       = lane
    emergency["hold_until"] = time.time() + EMERGENCY_HOLD_TIME

    if state["active_lane"] != lane or state["signal"] != "GREEN":
        state["active_lane"] = lane
        state["signal"]      = "GREEN"
        state["green_time"]  = MAX_GREEN
        state["timer_start"] = time.time()

    if not was_active:
        print(f"[EMERGENCY] *** AMBULANCE in {lane.upper()} — "
              f"GREEN priority for {MAX_GREEN}s + {EMERGENCY_HOLD_TIME}s hold ***")

def clear_emergency():
    """Ambulance no longer visible — start hold countdown."""
    if emergency["active"]:
        emergency["active"] = False
        hold_left = max(0, int(emergency["hold_until"] - time.time()))
        print(f"[EMERGENCY] Ambulance gone — holding GREEN for {hold_left}s more")

# ================================================================== #
#  DRAW HELPERS                                                        #
# ================================================================== #
def draw_emergency_banner(frame):
    in_emergency = emergency["active"] or time.time() < emergency["hold_until"]
    if not in_emergency:
        return
    # Flashing red banner
    color = (0, 0, 200) if flash_state else (0, 0, 80)
    cv2.rectangle(frame, (0, 0), (320, 22), color, -1)
    if emergency["active"]:
        txt = f"!! AMBULANCE — PRIORITY {state['active_lane'].upper()} !!"
    else:
        hold_left = max(0, int(emergency["hold_until"] - time.time()))
        txt = f"AMBULANCE CLEARED — HOLD {hold_left}s"
    cv2.putText(frame, txt, (4, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)

def draw_scene(frame):
    overlay = frame.copy()
    for lane, poly in LANE_POLYGONS.items():
        color = (0, 255, 0) if lane == state["active_lane"] else (0, 0, 255)
        cv2.fillPoly(overlay, [poly], color)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

    for lane, poly in LANE_POLYGONS.items():
        color = (0, 255, 0) if lane == state["active_lane"] else (0, 0, 255)
        cv2.polylines(frame, [poly], True, color, 2)
        M = cv2.moments(poly)
        if M["m00"] != 0:
            lx = int(M["m10"] / M["m00"])
            ly = int(M["m01"] / M["m00"])
            cv2.putText(frame, f"{lane}:{state['counts'][lane]}",
                        (lx - 20, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    draw_emergency_banner(frame)

    if CALIBRATE:
        for pt in calib_points:
            cv2.circle(frame, pt, 5, (255, 0, 255), -1)
        if len(calib_points) >= 2:
            for i in range(len(calib_points) - 1):
                cv2.line(frame, calib_points[i], calib_points[i+1], (255, 0, 255), 1)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and CALIBRATE:
        calib_points.append((x, y))
        print(f"[CALIB] ({x}, {y}) — total={len(calib_points)}")

# ================================================================== #
#  MAIN LOOP                                                           #
# ================================================================== #
print("[INFO] Running... Press q to quit")
update_leds()
cv2.namedWindow("Traffic")
cv2.setMouseCallback("Traffic", mouse_callback)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            check_signal_timing()
            update_leds()
            continue

        frame_display = cv2.resize(frame, (320, 240))
        frame_infer   = cv2.resize(frame, (640, 480))

        frame_count += 1
        check_signal_timing()
        update_leds()

        # ── Skip odd frames ──────────────────────────────────────── #
        if frame_count % 2 != 0:
            draw_scene(frame_display)
            cv2.imshow("Traffic", frame_display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # ── Reset tracked IDs every 60 s ─────────────────────────── #
        if time.time() - last_reset > 60:
            for lane in counted_ids:
                counted_ids[lane].clear()
            id_frame_count.clear()
            last_reset = time.time()
            print("[INFO] Tracked IDs reset.")

        # ── YOLO inference on 640×480 ─────────────────────────────── #
        results = model.track(
            frame_infer,
            persist=True,
            conf=0.10,
            iou=0.3,
            imgsz=640,
            verbose=False
        )[0]

        found_ambulance_this_frame = False

        for box in results.boxes:
            if box.id is None:
                continue

            track_id = int(box.id[0])
            cls_id   = int(box.cls[0])
            conf     = float(box.conf[0])

            if cls_id not in VEHICLE_CLASSES:
                continue

            # ── Scale box 640×480 → 320×240 for display ─────────── #
            x1, y1, x2, y2 = box.xyxy[0]
            x1d = int(x1 * 320 / 640)
            y1d = int(y1 * 240 / 480)
            x2d = int(x2 * 320 / 640)
            y2d = int(y2 * 240 / 480)

            if (x2d - x1d) * (y2d - y1d) < 80:
                continue

            cx, cy = (x1d + x2d) // 2, (y1d + y2d) // 2

            # Stability buffer
            id_frame_count[track_id] = id_frame_count.get(track_id, 0) + 1

            # ── Crop from HIGH-RES frame for colour check ─────────── #
            ix1 = int(x1); iy1 = int(y1)
            ix2 = int(x2); iy2 = int(y2)
            crop = frame_infer[
                max(0, iy1):min(480, iy2),
                max(0, ix1):min(640, ix2)
            ]

            # ── WHITE colour check = ambulance ───────────────────── #
            is_ambulance = (
                conf >= EMERGENCY_CONF
                and id_frame_count[track_id] >= STABLE_FRAMES
                and is_white_vehicle(crop)
            )

            # ── Lane assignment ───────────────────────────────────── #
            if id_frame_count[track_id] >= STABLE_FRAMES:
                for lane, poly in LANE_POLYGONS.items():
                    if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                        counted_ids[lane].add(track_id)

                        if is_ambulance:
                            found_ambulance_this_frame = True
                            trigger_emergency(lane)

            # ── Draw bounding box ────────────────────────────────── #
            if is_ambulance:
                # Bright white box + AMBULANCE label for emergency vehicle
                cv2.rectangle(frame_display, (x1d, y1d), (x2d, y2d), (255, 255, 255), 2)
                cv2.putText(frame_display, f"AMBULANCE {track_id}",
                            (x1d, y1d - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
            else:
                cv2.rectangle(frame_display, (x1d, y1d), (x2d, y2d), (0, 255, 255), 1)
                cv2.putText(frame_display,
                            f"{VEHICLE_CLASSES[cls_id]} {track_id} {conf:.2f}",
                            (x1d, y1d - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

            cv2.circle(frame_display, (cx, cy), 4, (0, 0, 255), -1)

            print(f"[DET] id={track_id} cls={VEHICLE_CLASSES[cls_id]} "
                  f"conf={conf:.2f} white={is_ambulance} center=({cx},{cy})")

        # ── Clear emergency if ambulance not seen this frame ─────── #
        if not found_ambulance_this_frame:
            clear_emergency()

        # ── Update counts ─────────────────────────────────────────── #
        state["counts"] = {lane: len(ids) for lane, ids in counted_ids.items()}

        check_signal_timing()
        update_leds()

        # ── Draw and show ─────────────────────────────────────────── #
        draw_scene(frame_display)
        cv2.imshow("Traffic", frame_display)

        remaining = max(0, state["green_time"] - (time.time() - state["timer_start"]))
        emg_str = f" [AMBULANCE→{emergency['lane']}]" if emergency["active"] else ""
        print(f"Lane2:{state['counts']['lane2']} Lane3:{state['counts']['lane3']} "
              f"-> {state['active_lane'].upper()} {state['signal']} "
              f"({int(remaining)}s left){emg_str}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and CALIBRATE and len(calib_points) >= 4:
            pts = calib_points[:4]
            print(f'\n[CALIB] "{calib_lane}": np.array([')
            for p in pts:
                print(f'    [{p[0]}, {p[1]}],')
            print('], dtype=np.int32)\n')
            calib_points.clear()

finally:
    cap.release()
    cv2.destroyAllWindows()
    for lane in LED_PINS:
        for color in LED_PINS[lane]:
            lgpio.gpio_write(CHIP, LED_PINS[lane][color], 0)
            lgpio.gpio_free(CHIP, LED_PINS[lane][color])
    lgpio.gpiochip_close(CHIP)
    print("[INFO] System stopped safely.")