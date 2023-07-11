import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

video = cv2.VideoCapture("video.mp4") #leer el video

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while True:
        check, frame = video.read()
        if not check:
            break

        # Convertir la imagen a RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Realizar la detección de la malla facial con MediaPipe
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Obtener los índices de los puntos clave de los labios
                upper_lip_indices = [61, 62, 63, 64, 65, 185, 92, 186]
                lower_lip_indices = [39, 40, 41, 42, 43, 44, 78, 95]

                # Dibujar los puntos clave de los labios
                for index in upper_lip_indices + lower_lip_indices:
                    landmark = face_landmarks.landmark[index]
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                # Dibujar las líneas que conectan los puntos clave de los labios
                for i in range(len(upper_lip_indices) - 1):
                    x1 = int(face_landmarks.landmark[upper_lip_indices[i]].x * frame.shape[1])
                    y1 = int(face_landmarks.landmark[upper_lip_indices[i]].y * frame.shape[0])
                    x2 = int(face_landmarks.landmark[upper_lip_indices[i + 1]].x * frame.shape[1])
                    y2 = int(face_landmarks.landmark[upper_lip_indices[i + 1]].y * frame.shape[0])
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                for i in range(len(lower_lip_indices) - 1):
                    x1 = int(face_landmarks.landmark[lower_lip_indices[i]].x * frame.shape[1])
                    y1 = int(face_landmarks.landmark[lower_lip_indices[i]].y * frame.shape[0])
                    x2 = int(face_landmarks.landmark[lower_lip_indices[i + 1]].x * frame.shape[1])
                    y2 = int(face_landmarks.landmark[lower_lip_indices[i + 1]].y * frame.shape[0])
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                upper_lip_distance = face_landmarks.landmark[61].y - face_landmarks.landmark[185].y
                lower_lip_distance = face_landmarks.landmark[39].y - face_landmarks.landmark[44].y

                if upper_lip_distance > 0.002 and lower_lip_distance > 0.002:
                    cv2.putText(frame, 'vocal abierta', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'Vocal cerrada', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        # Mostrar la imagen en una ventana
        cv2.imshow('MediaPipe FaceMesh', frame)

        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

