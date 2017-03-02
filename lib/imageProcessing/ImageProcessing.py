import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from skimage.util.dtype import dtype_range
from skimage.util.shape import view_as_windows

class detect:

    full_cascade = cv2.CascadeClassifier("/home/prema/opencv-3.1.0/data/haarcascades/haarcascade_fullbody.xml")
    face_cascade = cv2.CascadeClassifier("/home/prema/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml")
    os.system("espeak 'Pleasure welcoming you'")
    video_capture = cv2.VideoCapture(0)
    def Bgr_Gray(self,face_cascade,video_capture):
        #id=0
        while True:
            self.ret,self.frame = video_capture.read()
            self.gray = cv2.cvtColor(self.frame ,cv2.COLOR_BGR2GRAY)
            self.faces = face_cascade .detectMultiScale(self.gray,1.3,5)
            if len(self.faces) is 0:
              # execfile("/home/gowd95/PycharmProjects/project1/VIDEO/Audio_Alert.py")
                os.system("espeak ''")
            for (x,y,w,h) in self.faces:
                    cv2.rectangle(self.frame ,(x,y), (x+w, y+h),(0,255, 0),2)
                    cv2.imshow('Video', self.frame)
                    self.roi_gray = self.gray[y:y + h, x:x + w]
                    self.roi_color = self.frame[y:y + h, x:x + w]
                    #id = id + 1
                    # cv2.imwrite("/home/prema/Images4pro/User." + str(id) + " .png", self.roi_color)
                    cv2.imwrite("/home/prema/Images4pro/User.png", self.roi_color)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_capture.release()
        #cv2.destroyAllWindows()


    def ssim(self,X, Y, win_size=7,gradient=False, dynamic_range=None):
        if not X.dtype == Y.dtype:
            raise ValueError('Input images must have the same dtype.')

        if not X.shape == Y.shape:
            raise ValueError('Input images must have the same dimensions.')

        if not (win_size % 2 == 1):
            raise ValueError('Window size must be odd.')

        if dynamic_range is None:
            dmin, dmax = dtype_range[X.dtype.type]
            dynamic_range = dmax - dmin

        XW = view_as_windows(X, (win_size, win_size))
        YW = view_as_windows(Y, (win_size, win_size))

        NS = len(XW)
        NP = win_size * win_size

        ux = np.mean(np.mean(XW, axis=2), axis=2)
        uy = np.mean(np.mean(YW, axis=2), axis=2)

        # Compute variances var(X), var(Y) and var(X, Y)
        cov_norm = 1 / (win_size ** 2 - 1)
        XWM = XW - ux[..., None, None]
        YWM = YW - uy[..., None, None]
        vx = cov_norm * np.sum(np.sum(XWM ** 2, axis=2), axis=2)
        vy = cov_norm * np.sum(np.sum(YWM ** 2, axis=2), axis=2)
        vxy = cov_norm * np.sum(np.sum(XWM * YWM, axis=2), axis=2)

        R = dynamic_range
        K1 = 0.01
        K2 = 0.03
        C1 = (K1 * R) ** 2
        C2 = (K2 * R) ** 2

        A1, A2, B1, B2 = (v[..., None, None] for v in
                          (2 * ux * uy + C1,
                           2 * vxy + C2,
                           ux ** 2 + uy ** 2 + C1,
                           vx + vy + C2))

        S = np.mean((A1 * A2) / (B1 * B2))

        if gradient:
            local_grad = 2 / (NP * B1 ** 2 * B2 ** 2) * \
                         (
                             A1 * B1 * (B2 * XW - A2 * YW) - \
                             B1 * B2 * (A2 - A1) * ux[..., None, None] + \
                             A1 * A2 * (B1 - B2) * uy[..., None, None]
                         )

            grad = np.zeros_like(X, dtype=float)
            OW = view_as_windows(grad, (win_size, win_size))

            OW += local_grad
            grad /= NS

            return S, grad

        else:
            return S

    cv2.waitKey(0)
    cv2.destroyAllWindows()


face_obj=detect()
face_obj.Bgr_Gray(detect.face_cascade,detect.video_capture)
images = []
for img in glob.glob("/home/prema/Images4pro/*.png"):
    n= cv2.imread(img)
    images.append(n)
    img1 = cv2.imread(img)
    img2 = cv2.imread("/home/prema/Images4pro/User.png")

    img11 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img12 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    imageA = cv2.resize(img11, (100, 100))
    imageB = cv2.resize(img12, (100, 100))

    s = face_obj.ssim(imageA, imageB,win_size=7,gradient=False,dynamic_range=None)
    title = "Comparing"
    fig = plt.figure(title)
    if s < 0:
        s = 0
        plt.suptitle("Percentage : %.2f " % (s * 100))
    a=int(s*100)
    if a>50:
        os.system("espeak 'Successful match ,Thank you'")
        print(a)
        print("Successful Match,")
        print(img)






