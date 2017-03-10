import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure
import os

class detect:
    full_cascade = cv2.CascadeClassifier("/home/prema/opencv-3.1.0/data/haarcascades/haarcascade_fullbody.xml")
    face_cascade = cv2.CascadeClassifier('/home/prema/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml')
    os.system("espeak 'Pleasure welcoming you'")
    video_capture = cv2.VideoCapture(0)
    def Bgr_Gray(self,face_cascade,video_capture):
        while True:
            self.ret,self.frame = video_capture.read()
            self.gray = cv2.cvtColor(self.frame ,cv2.COLOR_BGR2GRAY)
            self.faces = face_cascade .detectMultiScale(self.gray,1.3,5)
            if len(self.faces) is 0:
                os.system("espeak 'i can not see you,please be in your position'")
            for (x,y,w,h) in self.faces:
                cv2.rectangle(self.frame ,(x,y), (x+w, y+h),(0,255, 0),2)
                cv2.imshow('Video', self.frame)
                self.roi_gray = self.gray[y:y + h, x:x + w]
                self.roi_color = self.frame[y:y + h, x:x + w]
                cv2.imwrite("/home/prema/Documents/Prems/d.jpg",self.roi_color)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_capture.release()
        cv2.destroyAllWindows()
    #def pixels(self):
    def histogram(self):
        self.image = color.rgb2gray(data.imread("/home/prema/Documents/Prems/aaa.jpg"))
        self.fd,self.hog_image = hog(self.image, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualise=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        #fig,(ax2)=plt.plot()
        hog_image_rescaled = exposure.rescale_intensity(self.hog_image, in_range=(0, 0.02))
        ax1.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        ax1.set_adjustable('box-forced')
        plt.show()


face_obj=detect()
face_obj.Bgr_Gray(detect.face_cascade,detect.video_capture)
face_obj.histogram()