import cv2
import os

# Student ID aur Name lena
student_id = input("Enter Student ID: ")
student_name = input("Enter Student Name: ")

# Student ke liye folder banana
folder_path = f"../dataset/{student_id}_{student_name}"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Camera start
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera frame nahi mila")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:
        count += 1

        # Face crop karna
        face = gray[y:y+h, x:x+w]

        # Image save karna
        file_path = f"{folder_path}/face_{count}.jpg"
        cv2.imwrite(file_path, face)

        # Face ke around rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Images: {count}/30",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("Student Registration", frame)

    # 30 images hone par stop
    if count >= 30:
        break

    # Q press karke manually stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nRegistration complete!")
print(f"{count} face images saved in: {folder_path}")