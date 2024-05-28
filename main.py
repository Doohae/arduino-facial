import cv2
import mediapipe as mp
import time
import serial

# Initialize mediapipe face detection and face mesh modules
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


# Initialize video capture
cap = cv2.VideoCapture(0)

# TODO: Replace 'COM3' with your Bluetooth serial port
BLUETOOTH_PORT = 'COM3'
# TODO: Adjust the eye aspect ratio threshold
EYE_THRESHOLD = 25
# TODO: facial expression and signal mapping
EXP_MAPPER = {
    "left_eye_closed": "L",
    "right_eye_closed": "R",
    "both_eyes_closed": "P",
    "smiling_closed_mouth": "F",
    "smiling_open_mouth": "B"
}

# Function to detect smiling without mouth opened based on the mouth aspect ratio
def is_smiling_closed_mouth(landmarks, img_width, img_height):
    left_mouth_corner = landmarks[61]
    right_mouth_corner = landmarks[291]
    upper_lip_top = landmarks[13]
    lower_lip_bottom = landmarks[14]

    horizontal_distance = abs(left_mouth_corner.x - right_mouth_corner.x) * img_width
    vertical_distance = abs(upper_lip_top.y - lower_lip_bottom.y) * img_height

    if horizontal_distance > vertical_distance * 2 and vertical_distance < 10:  # Adjust threshold for mouth openness
        return True
    return False

# Function to detect smiling with mouth opened based on the mouth aspect ratio
def is_smiling_open_mouth(landmarks, img_width, img_height):
    left_mouth_corner = landmarks[61]
    right_mouth_corner = landmarks[291]
    upper_lip_top = landmarks[13]
    lower_lip_bottom = landmarks[14]

    horizontal_distance = abs(left_mouth_corner.x - right_mouth_corner.x) * img_width
    vertical_distance = abs(upper_lip_top.y - lower_lip_bottom.y) * img_height

    if horizontal_distance > vertical_distance * 2 and vertical_distance >= 10:  # Adjust threshold for mouth openness
        return True
    return False

# Function to detect both eyes closed based on the eye aspect ratio
def are_both_eyes_closed(landmarks, img_height):
    left_eye_top = landmarks[159]
    left_eye_bottom = landmarks[145]
    right_eye_top = landmarks[386]
    right_eye_bottom = landmarks[374]

    left_eye_distance = abs(left_eye_top.y - left_eye_bottom.y) * img_height
    right_eye_distance = abs(right_eye_top.y - right_eye_bottom.y) * img_height

    if left_eye_distance < EYE_THRESHOLD and right_eye_distance < EYE_THRESHOLD:  # Adjusted threshold
        return True
    return False

# Function to detect only left eye closed based on the eye aspect ratio
def is_left_eye_closed(landmarks, img_height):
    left_eye_top = landmarks[159]
    left_eye_bottom = landmarks[145]

    left_eye_distance = abs(left_eye_top.y - left_eye_bottom.y) * img_height

    if left_eye_distance < EYE_THRESHOLD:  # Adjusted threshold
        return True
    return False

# Function to detect only right eye closed based on the eye aspect ratio
def is_right_eye_closed(landmarks, img_height):
    right_eye_top = landmarks[386]
    right_eye_bottom = landmarks[374]

    right_eye_distance = abs(right_eye_top.y - right_eye_bottom.y) * img_height

    if right_eye_distance < EYE_THRESHOLD:  # Adjusted threshold
        return True
    return False


def main():

    # Set up the serial connection (replace 'COM3' with your Bluetooth serial port)
    ser = serial.Serial(BLUETOOTH_PORT, 9600, timeout=1)
    time.sleep(2)  # Wait for the connection to initialize

    def send_command(command):
        try:
            ser.write(command.encode())
            print(f"Sent command: {command}")
        except Exception as e:
            print(f"Error: {e}")

    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()
            img_height, img_width, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_CONTOURS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
                    )

                    landmarks = face_landmarks.landmark

                    if is_left_eye_closed(landmarks, img_height) and not is_right_eye_closed(landmarks, img_height):
                        cv2.putText(frame, "Left Eye Closed", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        send_command(EXP_MAPPER["left_eye_closed"])
                        time.sleep(0.5)

                    elif is_right_eye_closed(landmarks, img_height) and not is_left_eye_closed(landmarks, img_height):
                        cv2.putText(frame, "Right Eye Closed", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        send_command(EXP_MAPPER["right_eye_closed"])
                        time.sleep(0.5)

                    elif are_both_eyes_closed(landmarks, img_height):
                        cv2.putText(frame, "Both Eyes Closed", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                        send_command(EXP_MAPPER["both_eyes_closed"])
                        time.sleep(0.5)

                    elif is_smiling_closed_mouth(landmarks, img_width, img_height):
                        cv2.putText(frame, "Smiling w/o Mouth Open", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        send_command(EXP_MAPPER["smiling_closed_mouth"])
                        time.sleep(0.5)

                    elif is_smiling_open_mouth(landmarks, img_width, img_height):
                        cv2.putText(frame, "Smiling w/ Mouth Open", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                        send_command(EXP_MAPPER["smiling_open_mouth"])
                        time.sleep(0.5)

                    else:
                        cv2.putText(frame, "Neutral", (50, 0), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

            end_time = time.time()
            fps = 1 / (end_time - start_time)
            cv2.putText(frame, f'FPS: {int(fps)}', (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    ser.close()


if __name__ == "__main__":
    main()
