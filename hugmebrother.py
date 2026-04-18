import cv2
import mediapipe as mp

def get_gesture(hand_landmarks, handedness):
    lm = hand_landmarks.landmark

    # Which fingers are extended
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

    # Direction: compare index fingertip x to wrist x
    # MediaPipe flips handedness when hand faces away from camera
    # so we use raw x difference for direction
    tip_x = lm[8].x
    wrist_x = lm[0].x
    diff = tip_x - wrist_x

    if diff > 0.1:
        return "POINTING LEFT"
    elif diff < -0.1:
        return "POINTING RIGHT"
    else:
        return ""  # pointing straight, not enough to one side



mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(index=0)

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
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                gesture = get_gesture(hand_landmarks, handedness)
                if gesture:
                    print(gesture)
                    cv2.putText(frame, gesture, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        cv2.imshow("Hand Tracking", cv2.flip(frame, 1))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv1.destroyAllWindows()

