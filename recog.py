import cv2
from model import load, _predict
from show import show_image

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    # 人脸识别特征分类器，系统自带
    cascade_path = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"

    model = load()

    while True:
        _, frame = cap.read()
        # 转换颜色空间为灰色
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 对象检测
        cascade = cv2.CascadeClassifier(cascade_path)
        # detectMultiScale函数。它可以检测出图片中所有的人脸，并将人脸用vector保存各个人脸的坐标、大小（用矩形表示)，可能有多个
        facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))
        if len(facerect) > 0:
            print('face detected')
            color = (255, 255, 255)  # 白
            for rect in facerect:

                x, y = rect[0:2]  # 前两项是坐标
                width, height = rect[2:4]  # 后两项是大小
                image = frame[y - 10: y + height, x: x + width]  # frame是整张图，只截取人脸的特定区域的图
                result = _predict(model, image)
                if result == 0:  # boss
                    print('someone is approaching')
                    show_image()
                else:
                    print('Nobody')

        k = cv2.waitKey(100)  # 输入27时，会中止程序
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
