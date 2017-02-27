import cv2
import numpy as np
import dlib
import io
import sys
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure
import pyttsx
import mutant.audioprocessing
class detect:
    PREDICTOR_PATH = "/home/gowd95/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    full_cascade = cv2.CascadeClassifier("/home/gowd95/opencv-3.1.0/data/haarcascades/haarcascade_fullbody.xml")
    face_cascade = cv2.CascadeClassifier("/home/gowd95/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml")
    engine = pyttsx.init()
    engine.say('Pleasure welcoming you')
    engine.runAndWait()
    video_capture = cv2.VideoCapture(0)
    def Bgr_Gray(self,face_cascade,video_capture):
        while True:
            self.ret,self.frame = video_capture.read()
            self.gray = cv2.cvtColor(self.frame ,cv2.COLOR_BGR2GRAY)
            self.faces = face_cascade .detectMultiScale(self.gray,1.3,5)
            if len(self.faces) is 0:
              # execfile("/home/gowd95/PycharmProjects/project1/VIDEO/Audio_Alert.py")
              engine = pyttsx.init()
              engine.say('i can not see you,please be in your position')
              engine.runAndWait()

            for (x,y,w,h) in self.faces:
                    cv2.rectangle(self.frame ,(x,y), (x+w, y+h),(0,255, 0),2)
                    cv2.imshow('Video', self.frame)
                    self.roi_gray = self.gray[y:y + h, x:x + w]
                    self.roi_color = self.frame[y:y + h, x:x + w]
                    cv2.imwrite("/home/gowd95/Desktop/gray1.jpg",self.roi_color)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_capture.release()
        #cv2.destroyAllWindows()

    def histogram(self):
        self.image = color.rgb2gray(data.imread("/home/gowd95/Desktop/gray1.jpg"))
        self.fd,self.hog_image = hog(self.image, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualise=True)
        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        hog_image_rescaled = exposure.rescale_intensity(self.hog_image, in_range=(0, 0.02))
        ax1.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        ax1.set_adjustable('box-forced')
        plt.show()

    def get_landmarks(self,face_cascade,predictor,im):
        self.rects = face_cascade.detectMultiScale(im, 1.3, 5)
        x, y, w, h = self.rects[0].astype(long)
        self.rect = dlib.rectangle(x, y, x + w, y + h)
        return np.matrix([[p.x, p.y] for p in predictor(im, self.rect).parts()])

    def annotate_landmarks(self,im, landmarks):
        self.im = im.copy()
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            cv2.putText(im, str(idx), pos,
                        fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        fontScale=0.4,
                        color=(0, 0, 255))
            cv2.circle(im, pos, 3, color=(0, 255, 255))
        return im
    #cv2.waitKey(0)

    def coordinate(self,predictor):
        file = sys.argv
        face_detector = dlib.get_frontal_face_detector()
        win = dlib.image_window()
        file = sys.argv
        image = cv2.imread('/home/gowd95/Desktop/1.jpg')

        detected_faces = face_detector(image, 1)
        print("Found {} faces in the image file {}".format(len(detected_faces), file))
        win.set_image(image)
        for i, face_rect in enumerate(detected_faces):
            print("- Face {} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.right(),face_rect.top(), face_rect.bottom()))
            win.add_overlay(face_rect)
            pose_landmarks = predictor(image, face_rect)
            win.add_overlay(pose_landmarks)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    dlib.hit_enter_to_continue()


face_obj=detect()
face_obj.Bgr_Gray(detect.face_cascade,detect.video_capture)
face_obj.histogram()
face_obj.coordinate(detect.predictor)
im= cv2.imread('/home/gowd95/Desktop/1.jpg')
res = face_obj.get_landmarks(detect.face_cascade, detect.predictor, im)
cv2.imshow('Recogn', face_obj.annotate_landmarks(im,res))





