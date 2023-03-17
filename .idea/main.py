import face_recognition
import imutils
import pickle
import time
import cv2
import os

cascPathface = os.path.dirname(cv2.__file__) + "haarcascade_frontalface_default.xml" #поиск файла
faceCascade = cv2.CascadeClassifier(cascPathface) #загрузка в классификатор
data = pickle.loads(open('face_enc', "rb").read())   # загрузка лиц и вложений, сохраненные в последнем файле
video_capture = cv2.VideoCapture(0) #цикл на кадрах
while True: # захват
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # преобразовать входной кадр из BGR в RGB
    encodings = face_recognition.face_encodings(rgb)# сравнения лица с лицом на входе
    names = []
    for encoding in encodings: # цикл по лицевым вложениям incase
        matches = face_recognition.compare_faces(data["encodings"], encoding) # проверяем, нашли ли мы совпадение
        name = "Unknown"
        if True in matches: #yfодим позиции, в которых мы получаем True
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs: # цикл по совпадающим индексам
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)
        names.append(name)
        for ((x, y, w, h), name) in zip(faces, names): # цикл по распознанным лицам
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # масштабируем координаты лица
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & amp, 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()