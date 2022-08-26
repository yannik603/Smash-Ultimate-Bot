import cv2
import nxbt
cap = cv2.VideoCapture(0)
# The device number might be 0 or 1 depending on the device and the webcam
while(True):
    ret, frame = cap.read()
    #frame = cv2.resize(frame, (48,48), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(frame.shape)
cap.release()
cv2.destroyAllWindows()
