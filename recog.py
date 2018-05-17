import cv2
from model import load, _predict
from show import show_image

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cascade_path = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"

    model = load()

    while True:
        _, frame = cap.read()

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cascade_path)
        facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))
        if len(facerect) > 0:
            print('face detected')
            color = (255, 255, 255)  # 白
            for rect in facerect:

                x, y = rect[0:2]
                width, height = rect[2:4]
                image = frame[y - 10: y + height, x: x + width]

                result = _predict(model, image)
                if result == 0:  # boss
                    print('someone is approaching')
                    show_image()
                else:
                    print('Nobody')

        k = cv2.waitKey(100)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
