import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Webcam detect nahi hua")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame read nahi hua")
        break

    print("Frame aa raha hai")

    cv2.imshow("Smart Classroom Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()