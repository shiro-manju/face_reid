import cv2
import dlib
import sys
import numpy as np


scaler = 0.5

#Initialize face detector and shape predictor
detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('model/shape_predictor_5_face_landmarks.dat')

face_detector = cv2.FaceDetectorYN.create("model/yunet.onnx", "", (0, 0))



#load video
#cap = cv2.VideoCapture('movie/boshi_kaburu.avi')
#cap = cv2.VideoCapture('movie/kuro_sanmei.avi')
#cap = cv2.VideoCapture('movie/shihuku.avi')


cap = cv2.VideoCapture(0) #内臓カメラ
#cap = cv2.VideoCapture(1) #USBカメラ


#Face recognition
def use_dlib(cap):
    while True:
        # read frame buffer from video
        ret, img = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            continue


        # resize frame
        img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
        ori = img.copy()

        # detect faces
        faces = detector(img)

        #例外処理　顔が検出されなかった時

        if len(faces) == 0:
            #print('no faces')
            img_rec = img

        for face in faces:

            # rectangle visualize
            img_rec = cv2.rectangle(img, pt1=(face.left(), face.top()), pt2=(face.right(), face.bottom()),
                                color=(255, 255, 255), lineType=cv2.LINE_AA, thickness=2)

            # landmark
            dlib_shape = landmark_predictor(img,face)
            shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

            #print(shape_2d.shape)

            for s in shape_2d:
                cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)


        #cv2.imshow('original', ori)
        cv2.imshow('img_rec', img_rec)


        if cv2.waitKey(1) == ord('q'):
            sys.exit(1)

    cv2.destroyAllWindows()

    
use_dlib(cap)