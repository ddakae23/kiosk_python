import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.spatial.distance import cosine
import rule_based_chatbot


visitor = 0


# CSV 파일 초기화 또는 불러오기
csv_file = 'visits_log.csv'
try:
    # converters를 사용하여 문자열을 numpy 배열로 변환합니다.
    visits_df = pd.read_csv(csv_file, converters={'Face_ID': lambda x: np.fromstring(x[1:-1], sep=' ')})
except FileNotFoundError:
    visits_df = pd.DataFrame(columns=['Face_ID', 'Last_Visit'])

# MTCNN과 InceptionResnetV1 초기화
mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# CLAHE 객체 생성
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# 웹캠 초기화
cap = cv2.VideoCapture(0)

# 유사도 임계값 설정
threshold = 0.3

def check(boxes):
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(x2, frame.shape[1]), min(y2, frame.shape[0])
            face = frame[y1:y2, x1:x2]
while True:
    ret, frame = cap.read()
    if not ret:
        print("웹캠에서 이미지를 가져올 수 없습니다.")
        break

    # 얼굴 검출 및 임베딩 계산
    boxes, _ = mtcnn.detect(frame)
    if boxes is not None:
        temp_size = 0
        temp_box = None
        temp_face = None
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(x2, frame.shape[1]), min(y2, frame.shape[0])
            face = frame[y1:y2, x1:x2]
            if temp_size<face.size:
                temp_face=face
                temp_box = box
        box = temp_box
        face = temp_face
            # 얼굴 영역이 유효한지 검사
        if face.size > 0:
            # CLAHE를 적용하여 얼굴 영역의 명암을 개선
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_clahe = clahe.apply(face_gray)
            face = cv2.cvtColor(face_clahe, cv2.COLOR_GRAY2BGR)

            face = cv2.resize(face, (160, 160))
            face_tensor = torch.from_numpy(face).float()
            face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0)
            face_tensor.div_(255).sub_(0.5).div_(0.5)
            emb = resnet(face_tensor).detach().numpy().flatten()

            # 임베딩을 사용하여 CSV 파일에서 가장 유사한 얼굴 찾기
            min_dist = threshold
            min_idx = None
            for idx, row in visits_df.iterrows():
                dist = cosine(emb, row['Face_ID'])
                if dist < min_dist:
                    min_dist = dist
                    min_idx = idx
            if min_idx is None:
                index=visitor
                visitor+=1
            else:
                index = min_idx
            rule_based_chatbot.customer(index)


            if min_idx is not None:
                visits_df.at[min_idx, 'Last_Visit'] = datetime.now().isoformat()
            else:
                new_row = pd.DataFrame([{'Face_ID': emb, 'Last_Visit': datetime.now().isoformat()}])
                visits_df = pd.concat([visits_df, new_row], ignore_index=True)


            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            face_id_str = f'FaceID: {min_idx}' if min_idx is not None else 'New Face'
            cv2.putText(frame, face_id_str, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)
            rule_based_chatbot.customer(min_idx)
    cv2.imshow('Webcam Face Detection', frame)

    visits_df.to_csv(csv_file, index=False)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()