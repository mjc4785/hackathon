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
    3. Point LEFT  → timeline scrolls left
       Point RIGHT → timeline scrolls right
"""

import asyncio
import threading
import cv2
import mediapipe as mp
import websockets
from websockets.server import serve

# ─── CONFIG ──────────────────────────────────────────────────────────────────

WS_HOST       = "localhost"
WS_PORT       = 8765
SCROLL_AMOUNT = 300        # pixels to scroll per gesture pulse
GESTURE_COOLDOWN = 0.15   # seconds between scroll pulses (prevents flooding)

# ─── SHARED STATE ────────────────────────────────────────────────────────────

connected_clients: set = set()
loop: asyncio.AbstractEventLoop = None   # set when the WS thread starts


# ─── WEBSOCKET SERVER ────────────────────────────────────────────────────────

async def ws_handler(websocket):
    """Accept a browser connection and keep it open."""
    connected_clients.add(websocket)
    print(f"[WS] Browser connected  (total: {len(connected_clients)})")
    try:
        await websocket.wait_closed()
    finally:
        connected_clients.discard(websocket)
        print(f"[WS] Browser disconnected (total: {len(connected_clients)})")


async def broadcast(message: str):
    """Send a message to every connected browser."""
    if connected_clients:
        await asyncio.gather(
            *[client.send(message) for client in connected_clients],
            return_exceptions=True,
        )


def send_gesture(gesture: str):
    """Thread-safe helper called from the CV thread."""
    if loop and connected_clients:
        asyncio.run_coroutine_threadsafe(broadcast(gesture), loop)


def start_ws_server():
    """Run the WebSocket server in its own thread + event loop."""
    global loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _run():
        async with serve(ws_handler, WS_HOST, WS_PORT):
            print(f"[WS] Server listening on ws://{WS_HOST}:{WS_PORT}")
            await asyncio.Future()   # run forever

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

    # Only index finger extended = pointing
    only_index = fingers["index"] and not any(
        [fingers["thumb"], fingers["middle"], fingers["ring"], fingers["pinky"]]
    )
    if not only_index:
        return ""

    tip_x   = lm[8].x
    wrist_x = lm[0].x
    diff    = tip_x - wrist_x

    if diff > 0.1:
        return "POINTING_LEFT"
    elif diff < -0.1:
        return "POINTING_RIGHT"
    return ""


# ─── MAIN LOOP ───────────────────────────────────────────────────────────────

def main():
    # Start WS server in background thread
    ws_thread = threading.Thread(target=start_ws_server, daemon=True)
    ws_thread.start()

    mp_hands   = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    cap        = cv2.VideoCapture(index=0)

    last_gesture   = ""
    last_sent_time = 0.0

    import time

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=4,
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

            gesture = ""

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
                        if now - last_sent_time >= GESTURE_COOLDOWN:
                            print(f"[GESTURE] {gesture}")
                            send_gesture(gesture)
                            last_sent_time = now

                        label = gesture.replace("_", " ")
                        cv2.putText(
                            frame, label, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3,
                        )

            # Show client count in corner
            status = f"Browsers: {len(connected_clients)}"
            cv2.putText(frame, status, (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Hand Tracking", cv2.flip(frame, 1))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
