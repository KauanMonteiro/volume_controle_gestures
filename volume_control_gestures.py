import cv2
import mediapipe as mp
import os
import platform
import math

def set_volume(level):
    percentage = int(level * 100)
    os_name = platform.system()
    if os_name == "Linux":
        os.system(f'pactl set-sink-volume @DEFAULT_SINK@ {percentage}%')
    elif os_name == "Windows":
        os.system(f'nircmd.exe setsysvolume {int(percentage * 655.35)}')
    else:
        print("Unsupported OS for volume adjustment")

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5,max_num_hands =1)
mp_drawing = mp.solutions.drawing_utils

if not cap.isOpened():
    print("Erro ao abrir a câmera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Não foi possível capturar o frame")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Convert normalized coordinates to pixel coordinates
            x_index, y_index = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
            x_thumb, y_thumb = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])

            # Draw a line between the index finger tip and the thumb tip
            cv2.line(frame, (x_index, y_index), (x_thumb, y_thumb), (0, 255, 0), 2)

            # Calculate distance between thumb and index finger tips
            distance = math.sqrt((x_index - x_thumb) ** 2 + (y_index - y_thumb) ** 2)

            # Volume adjustment based on distance
            max_distance, min_distance = 300, 30
            volume_level = max(0.0, min(1.0, (distance - min_distance) / (max_distance - min_distance)))
            set_volume(volume_level)

            # Display volume level on the frame
            cv2.putText(frame, f"Volume: {volume_level * 100:.1f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Camera', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
