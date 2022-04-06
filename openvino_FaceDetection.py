'''
参考: https://qiita.com/syukan3/items/a562961cd50f6fc07705
'''
import cv2
import copy
import os
import numpy as np
from openvino.inference_engine import IECore

iecore = IECore()
conf_th = 0.6

#file_name = 'movie/boshi_kaburu.avi'
file_name = 'movie/kuro_sanmei.avi'
#file_name = 'movie/shihuku.avi'

#save_path = 'face_img/kuro_sanmei'
save_path = 'face_mask_img/kuro_sanmei'
os.makedirs(save_path, exist_ok=True)

# モデルの読み込み（顔検出）
# パスは変更してください （上記「マスク検出モデル」で取得した学習済みモデルを使用）
model_path = 'model/face-detection-adas-0001.xml'
weights_path = 'model/face-detection-adas-0001.bin'
face_net = iecore.read_network(model=model_path, weights=weights_path)
face_exec_net = iecore.load_network(network=face_net, device_name='CPU', num_requests=2)


# モデルの読み込み（マスク検出）
# パスは変更してください （上記「マスク検出モデル」で取得した学習済みモデルを使用）
origin_path = 'model/'
mask_net = iecore.read_network(model=origin_path+'face_mask.xml', weights=origin_path+'face_mask.bin')
mask_exec_net = iecore.load_network(network=mask_net, device_name='CPU', num_requests=2)
mask_input_blob =  next(iter(mask_exec_net.inputs))
mask_output_blob = next(iter(mask_exec_net.outputs))


def write_face(frame_face_list, frame, cmid_list):
    for ind, frame_face in enumerate(frame_face_list):
        ret = cv2.imwrite('./{}/{}_{}_{}.png'.format(save_path, frame, ind, cmid_list[ind]), frame_face)
        assert ret, "出力に失敗"
    #[cv2.imwrite('face_img/{}_{}.jpg'.format(frame, ind), frame_face) for ind, frame_face in enumerate(frame_face_list)]

def main():
    #load video
    cap = cv2.VideoCapture(file_name)
    #cap = cv2.VideoCapture(1)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video = cv2.VideoWriter(filename='./movie/face_detection_kuro_sanmei.mp4', fourcc=fourcc, fps=fps, frameSize=(w, h))


    frame_count = 0
    # メインループ
    while cap.isOpened():
        ret, frame = cap.read()
        # Reload on error
        if ret == False:
            break
        
        visual_frame = copy.deepcopy(frame)

        # 顔検出用の入力データフォーマットへ変換
        img = cv2.resize(frame, (672, 384))
        
        img = img.transpose((2, 0, 1))    # HWC -> CHW
        img = np.expand_dims(img, axis=0) # 次元合せ

        # 顔検出 推論実行
        out = face_exec_net.infer(inputs={'data': img})

        # 出力から必要なデータのみ取り出し
        out = out['detection_out']
        out = np.squeeze(out) #サイズ1の次元を全て削除

        # 検出されたすべての顔領域に対して１つずつ処理
        frame_face_list = []
        cmid_list = []
        for detection in out:
            
            # print('¥n===================')
            # print(detection)

            # conf値の取得
            confidence = float(detection[2])

            # conf値が0.5より大きい場合のみ感情推論とバウンディングボックス表示
            if confidence > conf_th:
                # バウンディングボックス座標を入力画像のスケールに変換
                xmin = int(detection[3] * frame.shape[1])
                ymin = int(detection[4] * frame.shape[0])
                xmax = int(detection[5] * frame.shape[1])
                ymax = int(detection[6] * frame.shape[0])

            # 顔検出領域はカメラ範囲内に補正する。特にminは補正しないとエラーになる
                if xmin < 0:
                    xmin = 0
                if ymin < 0:
                    ymin = 0
                if xmax > frame.shape[1]:
                    xmax = frame.shape[1]
                if ymax > frame.shape[0]:
                    ymax = frame.shape[0]

                # 顔領域のみ切り出し
                frame_face = frame[ ymin:ymax, xmin:xmax ]
                frame_face_list.append(frame_face)
                if ymin <= 672/2:
                    cmid_list.append(0)
                else:
                    cmid_list.append(1)
                
                
                # マスク検出モデル 入力データフォーマットへ変換
                img = cv2.resize(frame_face, (224, 224))   # サイズ変更
                img = img.transpose((2, 0, 1))    # HWC > CHW
                img = np.expand_dims(img, axis=0) # 次元合せ

                # マスク検出 推論実行
                out = mask_exec_net.infer(inputs={mask_input_blob: img})

                # 出力から必要なデータのみ取り出し
                mask_out = out[mask_output_blob]
                mask_out = np.squeeze(mask_out) #不要な次元の削減

                # 文字列描画
                if int(mask_out) > 0:
                    display_text = 'With Face Mask'
                else:
                    display_text = 'No Face Mask'

                # 文字列描画
                #cv2.putText(frame, display_text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255, 255), 2)
                
                
                # バウンディングボックス表示
                cv2.rectangle(visual_frame, (xmin, ymin), (xmax, ymax), color=(240, 180, 0), thickness=3)

        # 顔画像を出力
        #write_face(frame_face_list, frame_count, cmid_list)

        # 画像表示
        cv2.imshow('frame', visual_frame)
        output_video.write(visual_frame)


        frame_count += 1

        # qキーが押されたら終了
        if cv2.waitKey(1) == ord('q'):
            break

    # 終了処理
    cap.release()
    output_video.release()
    cv2.destroyAllWindows()

if __name__ == ('__main__'):
    main()