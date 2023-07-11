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

        # Dibujar los puntos clave y las conexiones en la imagen
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Obtener las coordenadas de los labios
                lip_landmarks = face_landmarks.landmark[mp_face_mesh.FaceLandmark.UPPER_LIP_TOP:mp_face_mesh.FaceLandmark.LOWER_LIP_BOTTOM + 1]

                # Dibujar los puntos clave de los labios
                for landmark in lip_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                # Dibujar las líneas que conectan los puntos clave de los labios
                for i in range(len(lip_landmarks) - 1):
                    x1 = int(lip_landmarks[i].x * frame.shape[1])
                    y1 = int(lip_landmarks[i].y * frame.shape[0])
                    x2 = int(lip_landmarks[i + 1].x * frame.shape[1])
                    y2 = int(lip_landmarks[i + 1].y * frame.shape[0])
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                # Reconocer las vocales (a, e, i, o, u) basadas en la posición de los labios
                lip_distance = lip_landmarks[6].y - lip_landmarks[0].y

                if lip_distance > 0.03:
                    cv2.putText(frame, 'Vocal: A', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif lip_distance > -0.01:
                    cv2.putText(frame, 'Vocal: E', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif lip_distance > -0.06:
                    cv2.putText(frame, 'Vocal: I', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif lip_distance > -0.1:
                    cv2.putText(frame, 'Vocal: O', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'Vocal: U', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        # Mostrar la imagen en una ventana
        cv2.imshow('MediaPipe FaceMesh', frame)

        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

