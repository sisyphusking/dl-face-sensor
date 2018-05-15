import numpy as np
import cv2
import time


def gen_pic():

    cap = cv2.VideoCapture(0)  # 从摄像头中取得视频
    while (cap.isOpened()):
        # 读取帧摄像头
        ret, frame = cap.read()
        if ret == True:
            pic_name = str(int(round(time.time() * 1000))) + ".jpg"
            pic = 'data/' + pic_name
            cv2.imwrite(pic, frame)
            # 键盘按 Q 退出
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    gen_pic()