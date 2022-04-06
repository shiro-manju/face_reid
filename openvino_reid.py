'''
参考: https://dev.classmethod.jp/articles/person-reidentification/#toc-3
'''
import numpy as np
import time
import random
import cv2
from openvino.inference_engine import IECore
from load_model import Model
import copy

sim_dis_th = 10 #500
iou_tarcking_th = 0.6
cos_sim_th = 0.75

class PersonDetector(Model):

    def __init__(self, model_path, device, ie_core, threshold, num_requests):
        super().__init__(model_path, device, ie_core, num_requests, None)
        _, _, h, w = self.input_size
        self.__input_height = h
        self.__input_width = w
        self.__threshold = threshold

    def __prepare_frame(self, frame):
        initial_h, initial_w = frame.shape[:2]
        scale_h, scale_w = initial_h / float(self.__input_height), initial_w / float(self.__input_width)
        in_frame = cv2.resize(frame, (self.__input_width, self.__input_height))
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape(self.input_size)
        return in_frame, scale_h, scale_w

    def infer(self, frame):
        in_frame, _, _ = self.__prepare_frame(frame)
        result = super().infer(in_frame)

        detections = []
        height, width = frame.shape[:2]
        for r in result[0][0]:
            conf = r[2]
            if(conf > self.__threshold):
                x1 = int(r[3] * width)
                y1 = int(r[4] * height)
                x2 = int(r[5] * width)
                y2 = int(r[6] * height)
                detections.append([x1, y1, x2, y2, conf])
        return detections

class PersonReidentification(Model):

    def __init__(self, model_path, device, ie_core, threshold, num_requests):
        super().__init__(model_path, device, ie_core, num_requests, None)
        _, _, h, w = self.input_size
        self.__input_height = h
        self.__input_width = w
        self.__threshold = threshold

    def __prepare_frame(self, frame):
        initial_h, initial_w = frame.shape[:2]
        scale_h, scale_w = initial_h / float(self.__input_height), initial_w / float(self.__input_width)
        in_frame = cv2.resize(frame, (self.__input_width, self.__input_height))
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape(self.input_size)
        return in_frame, scale_h, scale_w

    def infer(self, frame):
        in_frame, _, _ = self.__prepare_frame(frame)
        result =  super().infer(in_frame)
        return np.delete(result, 1)

class LandmarksRegression(Model):

    def __init__(self, model_path, device, ie_core, threshold, num_requests):
        super().__init__(model_path, device, ie_core, num_requests, None)
        _, _, h, w = self.input_size
        self.__input_height = h
        self.__input_width = w
        self.__threshold = threshold

    def __prepare_frame(self, frame):
        initial_h, initial_w = frame.shape[:2]
        scale_h, scale_w = initial_h / float(self.__input_height), initial_w / float(self.__input_width)
        in_frame = cv2.resize(frame, (self.__input_width, self.__input_height))
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape(self.input_size)
        return in_frame, scale_h, scale_w

    def infer(self, frame):
        in_frame, _, _ = self.__prepare_frame(frame)
        result =  super().infer(in_frame)
        it = iter(list(result[0]))
        landmarks = [(i[0,0], j[0,0]) for i, j in zip(it, it)]
        return landmarks


class Tracker:
    def __init__(self):
        # 識別情報のDB
        self.identifysDb = None
        # 中心位置のDB
        self.center = []
    
    def __getCenter(self, person):
        x = person[0] - person[2]
        y = person[1] - person[3]
        return (x,y)

    def __getDistance(self, person, index):
        (x1, y1) = self.center[index]
        (x2, y2) = self.__getCenter(person)
        a = np.array([x1, y1])
        b = np.array([x2, y2])
        u = b - a
        return np.linalg.norm(u)

    def __getIOUDistance(self, persons, index):
        if len(persons) <= 1:
            return False
        bbox = np.array(persons[index])
        candidates = np.array([person for ind, person in enumerate(persons) if index != ind])
        bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
        candidates_tl = candidates[:, :2]
        candidates_br = candidates[:, :2] + candidates[:, 2:]

        tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
                np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
        br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
                np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
        wh = np.maximum(0., br - tl)

        area_intersection = wh.prod(axis=1)
        area_bbox = bbox[2:].prod()
        area_candidates = candidates[:, 2:].prod(axis=1)
        if ((area_intersection/(area_bbox + area_candidates - area_intersection)) > iou_tarcking_th).any():
            return True
        else:
            return False

    def __isOverlap(self, persons, index):
        [x1, y1, x2, y2] = persons[index]
        for i, person in enumerate(persons):
            if(index == i):
                continue
            if(max(person[0], x1) <= min(person[2], x2) and max(person[1], y1) <= min(person[3], y2)):
                return True
        return False

    def getIds(self, identifys, persons):
        if(identifys.size==0):
            return []
        if self.identifysDb is None:
            self.identifysDb = identifys
            for person in persons:
                self.center.append(self.__getCenter(person))
        
        print("input: {} DB:{}".format(len(identifys), len(self.identifysDb)))
        similaritys = self.__cos_similarity(identifys, self.identifysDb)
        similaritys[np.isnan(similaritys)] = 0
        ids = np.nanargmax(similaritys, axis=1)

        for i, similarity in enumerate(similaritys):
            persionId = ids[i]
            d = self.__getDistance(persons[i], persionId) #d: bboxの中心間距離
            print("persionId:{} {} distance:{}".format(persionId,similarity[persionId], d))
            if self.__getIOUDistance(persons, i):
                self.identifysDb[persionId] += identifys[i]

            # 0.95以上で、重なりの無い場合、識別情報を更新する
            elif similarity[persionId] > cos_sim_th:
                if(self.__isOverlap(persons, i) == False):
                    self.identifysDb[persionId] += identifys[i]
            # 0.5以下で、距離が離れている場合、新規に登録する
            elif(similarity[persionId] < 0.5):
                if(d > sim_dis_th):
                    print("distance:{} similarity:{}".format(d, similarity[persionId]))
                    self.identifysDb = np.vstack((self.identifysDb, identifys[i]))
                    self.center.append(self.__getCenter(persons[i]))
                    ids[i] = len(self.identifysDb) - 1
                    print("> append DB size:{}".format(len(self.identifysDb)))

        print(ids)
        # 重複がある場合は、信頼度の低い方を無効化する
        for i, a in enumerate(ids):
            for e, b in enumerate(ids):
                if(e == i):
                    continue
                if(a == b):
                    if(similarity[a] > similarity[b]):
                        ids[i] = -1
                    else:
                        ids[e] = -1
        print(ids)
        return ids

    # コサイン類似度
    # 参考にさせて頂きました: https://github.com/kodamap/person_reidentification
    def __cos_similarity(self, X, Y):
        m = X.shape[0]
        Y = Y.T
        return np.dot(X, Y) / (
            np.linalg.norm(X.T, axis=0).reshape(m, 1) * np.linalg.norm(Y, axis=0)
        )
    # ユークリッド距離
    def __euclidean_similarity(self, X, Y):
        return 0

def align_face(face_frame, landmarks):

    left_eye, right_eye, tip_of_nose, left_lip, right_lip = landmarks
    # compute the angle between the eye centroids
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.arctan2(dy, dx) * 180 / np.pi

    # center of face_frame
    center = (face_frame.shape[0] // 2, face_frame.shape[1] // 2)
    h, w, c = face_frame.shape

    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    aligned_face = cv2.warpAffine(face_frame, M, (w, h))

    #visual_aligned_face = copy.deepcopy(aligned_face)
    #[cv2.circle(visual_aligned_face, (int(landmark[0]*w), int(landmark[1]*h)), 2, (255, 241, 0), thickness=-1) for landmark in landmarks]
    #cv2.imwrite("./test.png", visual_aligned_face)

    return aligned_face

def main():
    # face model
    face_model = True
    person_detector_model_path = './openvino_models/intel/face-detection-adas-0001/FP32/face-detection-adas-0001'
    person_reidentification_model_path = './openvino_models/intel/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095'
    landmarks_regression_model_path = './openvino_models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009'

    # person model
    # face_model = False
    # person_detector_model_path = './openvino_models/intel/person-detection-retail-0013/FP16/person-detection-retail-0013'
    # person_reidentification_model_path = './openvino_models/intel/face-reidentification-retail-0095/FP16-INT8/face-reidentification-retail-0095'

    device = "CPU"
    cpu_extension = None
    ie_core = IECore()
    if device == "CPU" and cpu_extension:
        ie_core.add_extension(cpu_extension, "CPU")

    THRESHOLD= 0.4
    person_detector = PersonDetector(person_detector_model_path, device, ie_core, THRESHOLD, num_requests=2)
    personReidentification = PersonReidentification(person_reidentification_model_path, device, ie_core, THRESHOLD, num_requests=2)
    tracker = Tracker()
    landmarks_regression = LandmarksRegression(landmarks_regression_model_path, device, ie_core, THRESHOLD, num_requests=2)



    #MOVIE = "./video001.mp4"
    MOVIE_NAME = 'boshi_kaburu'
    #MOVIE_NAME = 'kuro_sanmei'

    MOVIE = "./movie/{}.avi".format(MOVIE_NAME)
    SCALE = 2

    # Load video
    cap = cv2.VideoCapture (MOVIE)

    # Output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video = cv2.VideoWriter(filename='./movie/face_reid_{}.mp4'.format(MOVIE_NAME), fourcc=fourcc, fps=fps, frameSize=(int(SCALE*w), int(SCALE*h)))

    TRACKING_MAX = 50
    colors = []
    for i in range(TRACKING_MAX):
        b = random.randint(0, 255)
        g = random.randint(0, 255)
        r = random.randint(0, 255)
        b = 50
        g = 200
        r = 50
        colors.append((b,g,r))

    while True:
            
        grabbed, frame = cap.read()
        if not grabbed:# ループ再生
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # continue
            break
        if(frame is None):
            continue

        # Personを検知する
        persons = []
        detections =  person_detector.infer(frame)
        if(len(detections) > 0):
            print("-------------------")
            for detection in detections:
                x1 = int(detection[0])
                y1 = int(detection[1])
                x2 = int(detection[2])
                y2 = int(detection[3])
                conf = detection[4]
                print("{:.1f} ({},{})-({},{})".format(conf, x1, y1, x2, y2))
                persons.append([x1,y1,x2,y2])

        print("====================")
        # 各Personの画像から識別情報を取得する
        identifys = np.zeros((len(persons), 255))
        for i, person in enumerate(persons):
            # 各Personのimage取得
            img = frame[person[1] : person[3], person[0]: person[2]]
            h, w = img.shape[:2]
            if(h==0 or w==0):
                continue
            # landmarksの取得
            if face_model:
                landmarks = landmarks_regression.infer(img)
                if len(landmarks) == 5:
                    print("landmarks:", landmarks)
                    img = align_face(img, landmarks)

            # identification取得
            identifys[i] = personReidentification.infer(img)

        # Idの取得
        ids = tracker.getIds(identifys, persons)
        
        # 枠及びIdを画像に追加
        for i, person in enumerate(persons):
            if(ids[i]!=-1):
                color = colors[int(ids[i])]
                frame = cv2.rectangle(frame, (person[0], person[1]), (person[2] ,person[3]), color, int(2))
                frame = cv2.putText(frame, 'ID:{}'.format(ids[i]),  (person[0], person[1]), cv2.FONT_HERSHEY_PLAIN, int(2), color, int(2), cv2.LINE_AA )

        # 画像の縮小
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, ((int(w*SCALE), int(h*SCALE))))
        # 画像の表示

        cv2.imshow('frame', frame)
        output_video.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    output_video.release()
    cv2.destroyAllWindows()


if __name__==('__main__'):
    main()