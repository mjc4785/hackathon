"""
hand_tracker.py
---------------
Tracks hand gestures via webcam and broadcasts scroll commands
to any connected browser over WebSocket.

Requirements:
    pip install opencv-python mediapipe websockets

Usage:
    1. Run this script:   python hand_tracker.py
    2. Open timeline.html in your browser
    3. Point LEFT  → timeline scrolls left  (more fingers = faster)
       Point RIGHT → timeline scrolls right (more fingers = faster)
       Open palm   → highlight nearest event (hover)
"""

import asyncio
import threading
import time
import cv2
import mediapipe as mp
import websockets
from websockets.server import serve

# ─── CONFIG ──────────────────────────────────────────────────────────────────

WS_HOST          = "localhost"
WS_PORT          = 8765
GESTURE_COOLDOWN = 0.12   # seconds between scroll pulses

# ─── SHARED STATE ────────────────────────────────────────────────────────────

connected_clients: set = set()
loop: asyncio.AbstractEventLoop = None


# ─── WEBSOCKET SERVER ────────────────────────────────────────────────────────

async def ws_handler(websocket):
    connected_clients.add(websocket)
    print(f"[WS] Browser connected  (total: {len(connected_clients)})")
    try:
        await websocket.wait_closed()
    finally:
        connected_clients.discard(websocket)
        print(f"[WS] Browser disconnected (total: {len(connected_clients)})")


async def broadcast(message: str):
    if connected_clients:
        await asyncio.gather(
            *[client.send(message) for client in connected_clients],
            return_exceptions=True,
        )


def send_gesture(gesture: str):
    if loop and connected_clients:
        asyncio.run_coroutine_threadsafe(broadcast(gesture), loop)


def start_ws_server():
    global loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _run():
        async with serve(ws_handler, WS_HOST, WS_PORT):
            print(f"[WS] Server listening on ws://{WS_HOST}:{WS_PORT}")
            await asyncio.Future()

    loop.run_until_complete(_run())


# ─── GESTURE DETECTION ───────────────────────────────────────────────────────

def get_gesture(hand_landmarks, handedness):
    lm = hand_landmarks.landmark

    fingers = {
        "thumb":  lm[4].y  < lm[3].y,
        "index":  lm[8].y  < lm[6].y,
        "middle": lm[12].y < lm[10].y,
        "ring":   lm[16].y < lm[14].y,
        "pinky":  lm[20].y < lm[18].y,
    }

    # ── HOVER: all 5 fingers extended = open palm ──────────────────────────
    if all(fingers.values()):
        return "HOVER"

    # ── SCROLL: count extended non-thumb fingers ────────────────────────────
    # Thumb must be closed (fist-like base) to avoid false positives
    if fingers["thumb"]:
        return ""

    extended = [fingers["index"], fingers["middle"],
                fingers["ring"],  fingers["pinky"]]
    num = sum(extended)
    if num == 0:
        return ""

    # Average x of extended fingertips vs wrist
    tip_xs = [lm[8].x, lm[12].x, lm[16].x, lm[20].x]
    avg_tip_x = sum(x for x, ext in zip(tip_xs, extended) if ext) / num
    diff      = avg_tip_x - lm[0].x  # wrist

    if diff > 0.1:
        return f"POINTING_LEFT_{num}"   # 1–4 fingers
    elif diff < -0.1:
        return f"POINTING_RIGHT_{num}"
    return ""


# ─── MAIN LOOP ───────────────────────────────────────────────────────────────

# Labels shown in the OpenCV window
GESTURE_LABELS = {
    "HOVER": "HOVER",
    **{f"POINTING_LEFT_{n}":  f"<< LEFT  x{n}" for n in range(1, 5)},
    **{f"POINTING_RIGHT_{n}": f"RIGHT >> x{n}" for n in range(1, 5)},
}


def main():
    ws_thread = threading.Thread(target=start_ws_server, daemon=True)
    ws_thread.start()

    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    cap        = cv2.VideoCapture(index=0)

    last_sent_time = 0.0

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame...")
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = hands.process(frame_rgb)
            gesture   = ""

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                    gesture = get_gesture(hand_landmarks, handedness)

                    if gesture:
                        now = time.monotonic()
                        # HOVER sends continuously so browser can track it;
                        # scroll gestures respect cooldown
                        is_scroll = gesture.startswith("POINTING")
                        if not is_scroll or (now - last_sent_time >= GESTURE_COOLDOWN):
                            send_gesture(gesture)
                            if is_scroll:
                                last_sent_time = now

                        label = GESTURE_LABELS.get(gesture, gesture)
                        cv2.putText(frame, label, (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # Client count overlay
            cv2.putText(frame, f"Browsers: {len(connected_clients)}",
                        (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Hand Tracking", cv2.flip(frame, 1))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
