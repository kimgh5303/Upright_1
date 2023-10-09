import cv2
import argparse
import numpy as np
import socket
import struct
import pickle

# PoseNet 모델 로딩 및 설정
parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')

args = parser.parse_args()

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],                               # 0,    1,     2
                ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],                                 # 3,    4,     5
                ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],                          # 6,    7,     8,     9
                ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],                          # 10,   11,    12,    13
                ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"], ["Nose", "RShoulder"], ["Nose", "LShoulder"]]  # 14,   15,    16,    17,    18

inWidth = args.width
inHeight = args.height
net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")
        
# 서버 ip 주소 및 port 번호
ip = '210.125.31.236'
port = 3306
# 소켓 객체 생성
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 소켓 주소 정보 할당
server_socket.bind((ip, port))

# 연결 리스닝(동시 접속) 수 설정
server_socket.listen(10)
print('클라이언트 연결 대기')

while True:
    # 클라이언트 연결 대기
    client_socket, address = server_socket.accept()
    print('클라이언트 ip 주소:', address[0])

    try:
        while True:
            # 데이터 크기 수신
            data_size = client_socket.recv(4)
            if not data_size:
                break
            data_size = struct.unpack(">L", data_size)[0]

            # 이미지 데이터 수신
            frame_data = b""
            while len(frame_data) < data_size:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                frame_data += chunk

            # 이미지 데이터 디코딩
            frame_encoded = pickle.loads(frame_data)
            # frame_encoded = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(frame_encoded, cv2.IMREAD_COLOR)

            # 프레임을 읽어올 수 있는 경우에만 아래 코드를 실행
            if frame is not None and len(frame) > 0:
                frameWidth = frame.shape[1]     # 640
                frameHeight = frame.shape[0]    # 480

                net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
                out = net.forward()
                out = out[:, :19, :, :]

                assert(len(BODY_PARTS) == out.shape[1])

                points = []
                for i in range(len(BODY_PARTS)):
                    heatMap = out[0, i, :, :]
                    _, conf, _, point = cv2.minMaxLoc(heatMap)
                    x = (frameWidth * point[0]) / out.shape[3]
                    y = (frameHeight * point[1]) / out.shape[2]
                    points.append((int(x), int(y)) if conf > args.thr else None)

                # 포즈 데이터를 직렬화하여 클라이언트로 전송
                points_data = pickle.dumps(points)
                # 전송할 데이터의 크기 계산
                points_size = struct.pack(">L", len(points_data))
                # 데이터 전송
                client_socket.sendall(points_size + points_data)
            else:
                print("클라이언트 연결 종료")
                break

    except ConnectionResetError:
        print("클라이언트 연결 종료2")
    finally:
        client_socket.close()
