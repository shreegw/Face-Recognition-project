import cv2
import face_recognition
import numpy as np
from datetime import datetime

now = datetime.now()
file_open = open("Attendance.csv","w")
file_open.write("S.No, Roll Number, Name, Date")
file_open.write("\n")
sno=1

def known_face(sno, rno, name, date):
    row_string = "{},{},{},{}".format(sno, rno, name, date)
    file_open.write(row_string)
    file_open.write("\n")
    sno+=1

def unknown_face(sno, date, name="unknown", rno="N/A"):
    row_string = "{},{},{},{}".format(sno, rno, name, date)
    file_open.write(row_string)
    file_open.write("\n")
    sno+=1

def main_content(i):
    known_faces = {"Shree Ganesh": "17-459", "Harshith.B": "17-409", "D.Anvesh": "17-419"}
    date = now.strftime("%d/%m/%Y")
    if i in known_faces:
        known_face(sno, known_faces[i], i, date)
    else:
        unknown_face(sno, date)

video_capture = cv2.VideoCapture(0)

SG_image = face_recognition.load_image_file("SG.jpg")
SG_face_encoding = face_recognition.face_encodings(SG_image)[0]

Anvesh_image=face_recognition.load_image_file("anvesh.jpeg")
Anvesh_face_encoding=face_recognition.face_encodings(Anvesh_image)[0]

Harshith_image = face_recognition.load_image_file("Harshith.jpg")
Harshith_face_encoding = face_recognition.face_encodings(Harshith_image)[0]

known_face_encodings = [SG_face_encoding,Harshith_face_encoding,Anvesh_face_encoding]
known_face_names = ["Shree Ganesh","Harshith.B","D.Anvesh"]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)
            for i in face_names:
                main_content(i)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
