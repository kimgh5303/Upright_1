# To use Inference Engine backend, specify location of plugins:
# export LD_LIBRARY_PATH=/opt/intel/deeplearning_deploymenttoolkit/deployment_tools/external/mklml_lnx/lib:$LD_LIBRARY_PATH
import cv2 as cv
import numpy as np
import argparse
import math
import dlib
import Upright_DB
import User
import UserFig
import Ui
import time
from pymysql import NULL
from tkinter import messagebox
from scipy.spatial import distance as dist
from imutils import face_utils

running = False                             #웹캠 실행, 끄기
sensitivity = 1

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth, inner_mouth):
    A = dist.euclidean(mouth[10], inner_mouth[7])
    B = dist.euclidean(mouth[9], inner_mouth[6])
    C = dist.euclidean(mouth[8], inner_mouth[5])
    D = dist.euclidean(mouth[7], mouth[11])

    mar = (A+B+C) / (3.0 * D)
    return mar

def Save_Cam():
    now = time.localtime()
    name = str(now.tm_year)+str(now.tm_mon)+str(now.tm_mday)+str("_")+str(now.tm_hour)+str("_")+str(now.tm_min)+str("_")+str(now.tm_sec)
    return name

def camStart():
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

    n=range(20)
    n=range(20)

    inWidth = args.width
    inHeight = args.height
    net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

    cap = cv.VideoCapture(args.input if args.input else 0)    
    # cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv.CAP_PROP_FRAME_WIDTH, 900)

    global clickNumber, firstPose, badPose, poseStack_H, poseStack_V, poseStack_E, poseStack_M
    clickNumber = -1                                            #보정, 실행 버튼 클릭 처리
    index=-1                                                    #순서
    firstPose = badPose = poseStack_H = poseStack_V = poseStack_T = poseStack_E = poseStack_M = 0
    #처음 보정값, 스크린샷 스택 초기 값, 기울기 스택 초기 값
    rightShoulder_xcor = rightShoulder_ycor = leftShoulder_xcor = leftShoulder_ycor = 0
    #오른쪽 어깨 x,y좌표,                      왼쪽 어깨 x,y좌표
    bodyGradient_H = 0                                          #수평 기울기
    bodyGradient_V = 0                                          #수직 기울기
    percentage = 0

    userH = userV = userR = userL = userE = userM = NULL

    newORold: str
    BODY_LENGTH = [j*0 for j in n]                              #점 사이 선 길이
    img = [k*0 for k in n]

    userFig = Upright_DB.Information_DB("first")          #DB에서 불러온 정자세 보정 정보를 객체에 담아놓음
    # 유저 보정값 가져옴
    if (len(userFig)>0):UserFig.userFig.Set_User(userFig[0], userFig[1], userFig[2], userFig[3], userFig[4], userFig[5])
    else:UserFig.userFig.Set_User(NULL, NULL, NULL, NULL, NULL, NULL)

    userLearn = Upright_DB.Information_DB("second")          #DB에서 불러온 흐트러짐 정보를 객체에 담아놓음
    # 유저 측정값 가져옴
    if (len(userLearn)>0):UserFig.userLearn.Set_User(userLearn[0], userLearn[1], userLearn[2], userLearn[3], userLearn[4], userLearn[5])
    else:UserFig.userLearn.Set_User(NULL, NULL, NULL, NULL, NULL, NULL)

    #---------------------------------------------------------------
    firstEar=firstMar=0

    EYE_AR_CONSEC_FRAMES = 3

    EYECOUNTER = 0
    TOTAL = 0

    MTH_AR_THRESH = 0.26

    MOUTHCOUNTER = 0


    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')



    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

    (oStart, oEnd) = face_utils.FACIAL_LANDMARKS_IDXS['mouth']
    (iStart, iEnd) = face_utils.FACIAL_LANDMARKS_IDXS['inner_mouth']

    # 무한루프
    #---------------------------------------------------------------

    while running:                                              #루프 반복
        hasFrame, frame = cap.read()                            #웹캠으로부터 영상을 하나 가져옴

        if not hasFrame:                                        #웹캠으로부터 영상을 가져올 수 없으면 루프 중지
            cv.waitKey()
            break

        frameWidth = frame.shape[1]     #640
        frameHeight = frame.shape[0]    #480

        net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))             #네트워크 입력을 지정
        out = net.forward()                                                                                                             #네트워크의 출력을 얻기 위해 포워드 패스를 수행
        out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

        assert(len(BODY_PARTS) == out.shape[1])

        points = []                                                                                        #검출된 결과를 저장하기 위한 리스트를 정의
        for i in range(len(BODY_PARTS)):                                                                   #몸을 구성하는 모든 부분
            # Slice heatmap of corresponging body's part.
            heatMap = out[0, i, :, :]                                                                      #히트맵을 가짐
            _, conf, _, point = cv.minMaxLoc(heatMap)                                                      #최대값(conf)과 최대값의 위치(point)를 찾음
            
            #원본 영상에 맞게 좌표를 조정 -> 앞에서 이미지 크기를 줄였기 때문
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            # Add a point if it's confidence is higher than threshold.
            points.append((int(x), int(y)) if conf > args.thr else None)                                   #리스트에 해당 좌표를 입력

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # 분류할 이미지를 grayscale로 로드
        rects = detector(gray, 0)

# 동작 인식-----------------------------------------------------------------------------------------------------------------------------------------------                
        for pair in POSE_PAIRS:
            index=index+1
            if index>18:
                index=0
            partFrom = pair[0]
            partTo = pair[1]
            assert(partFrom in BODY_PARTS)              #bodyparts 안에 있나
            assert(partTo in BODY_PARTS)

            idFrom = BODY_PARTS[partFrom]
            idTo = BODY_PARTS[partTo]

            if points[idFrom] and points[idTo]:
                img[index] = cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)            #타원
                cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                BODY_LENGTH[index]=math.sqrt((points[idFrom][0]-points[idTo][0])**2+(points[idFrom][1]-points[idTo][1])**2)

                # 어깨 좌표 비교를 위해 값 저장
                if (index == 0):
                    rightShoulder_xcor = points[idTo][0]
                    rightShoulder_ycor = points[idTo][1]
                if (index == 1):
                    leftShoulder_xcor = points[idTo][0]
                    leftShoulder_ycor = points[idTo][1]
                try:
                    bodyGradient_H = ((rightShoulder_ycor - leftShoulder_ycor)/(rightShoulder_xcor - leftShoulder_xcor))
                except ZeroDivisionError:
                    continue
                
                # 코-목 수직 기울기
                if index == 12:
                    try:
                        bodyGradient_V = ((points[idFrom][1]-points[idTo][1])/(points[idFrom][0]-points[idTo][0]))       # y2-y1/x2-x1
                    except ZeroDivisionError:
                        continue
                
                # 정자세 보정
                if(clickNumber == 0):
                    cv.putText(frame, "Mode: {}".format("Correction"), (0, 30), cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
                    if(firstPose == 0):
                        if UserFig.userFig.bodyGradient_H==NULL: 
                            MsgBox = messagebox.askquestion("알림", "정자세 보정을 하겠습니까?")
                            newORold = "new"
                            if MsgBox == 'yes':
                                firstPose  = 1
                            else:
                                clickNumber = -1
                                break
                        else:
                            MsgBox = messagebox.askquestion("알림", "정자세 보정을 다시 하겠습니까?")
                            newORold = "old"
                            if MsgBox == 'yes':
                                firstPose += 1
                            else:
                                clickNumber = -1
                                break                            
                    if(firstPose == 1):                     
                        messagebox.showinfo("주의", "정자세를 해주세요")
                        messagebox.showinfo("주의", "눈을 감지 말고 입을 벌리지 마세요")
                        firstPose = 2
                    if(firstPose == 2):
                        cv.putText(frame, "Status: {}".format("Waiting..."), (0, 60), cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
                    if(firstPose == 3):
                        Upright_DB.SaveBodyDB(bodyGradient_H, bodyGradient_V, BODY_LENGTH[17], BODY_LENGTH[18], newORold)
                        firstPose = 4
                
                # 흐트러짐 측정
                if(clickNumber == 2):
                    if UserFig.userFig.bodyGradient_H==NULL: 
                        MsgBox = messagebox.askquestion("알림", "정자세 보정을 해주십시오")
                        clickNumber = -1
                        break
                    cv.putText(frame, "Mode: {}".format("Record"), (0, 30), cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
                    if(badPose == 0):
                        if UserFig.userLearn.bodyGradient_H==NULL: 
                            MsgBox = messagebox.askquestion("알림", "흐트러짐 측정을 하겠습니까?")
                            newORold = "new"
                            if MsgBox == 'yes':
                                badPose = 1
                            else:
                                clickNumber = -1
                                break
                        else:
                            MsgBox = messagebox.askquestion("알림", "흐트러짐 측정을 다시 하겠습니까?")
                            newORold = "old"
                            if MsgBox == 'yes':
                                badPose  = 1
                            else:
                                clickNumber = -1
                                break                              
                    if(badPose == 1):
                        messagebox.showinfo("알림", "흐트러진 자세를 측정합니다\n" "안내사항에 따라주세요")
                        messagebox.showinfo("알림", "몸 기울기를 측정합니다\n" "몸을 옆으로 기울여주세요")
                        badPose = 2
                    if(badPose==2):
                        cv.putText(frame, "Status: {}".format("Gradient Waiting..."), (0, 60), cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
                    if(badPose==3):
                            userH = abs(abs(bodyGradient_H)-abs(UserFig.userFig.bodyGradient_H))
                            userV = round(bodyGradient_V/(UserFig.userFig.bodyGradient_V),2)
                            messagebox.showinfo("완료", "몸 기울기 측정이 완료되었습니다")
                            messagebox.showinfo("알림", "거북목을 측정합니다\n" "얼굴을 앞으로 내밀어주세요")
                            badPose = 4
                    if(badPose==4):
                        cv.putText(frame, "Status: {}".format("Turtle Waiting..."), (0, 60), cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
                    if(badPose==5):
                        userR = round(BODY_LENGTH[17]/(UserFig.userFig.bodyLength_R),2)
                        userL = round(BODY_LENGTH[18]/(UserFig.userFig.bodyLength_L),2)
                        messagebox.showinfo("완료", "거북목 측정이 완료되었습니다")
                        messagebox.showinfo("알림", "눈을 측정합니다\n" "눈을 찡그려주세요")
                        badPose = 6
                        
                print(BODY_LENGTH[17])
                if(clickNumber == 1):
                    if UserFig.userFig.bodyGradient_H==NULL:
                        messagebox.showinfo("경고", "정자세 보정값과 흐트러짐을 잡아주세요")
                        clickNumber = -1
                        break
                    if UserFig.userLearn.bodyGradient_H==NULL: 
                        messagebox.showinfo("경고", "처음 흐트러짐을 잡아주세요")
                        clickNumber = -1
                        break
                    
                    # 몸이 기울었을때 -> 양쪽 어깨 좌표의 기울기로 측정
                    if(abs(abs(bodyGradient_H)-abs(UserFig.userFig.bodyGradient_H)) > abs(UserFig.userLearn.bodyGradient_H*(2-sensitivity))):
                        cv.putText(frame, "Gradient_H: {}".format("bad"), (420, 30), cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
                        poseStack_H += 1
                        if(poseStack_H == 200):
                            Upright_DB.Count_Habit("gradient")
                            path = "C:/imgfile/" + str("habit_") + Save_Cam() + ".png"
                            cv.imwrite(path, frame)
                            Upright_DB.insertBLOB(path)
                            messagebox.showinfo("경고", "자세가 기울었어요")
                            poseStack_H = 0
                            poseStack_V = 0
                    else:
                        cv.putText(frame, "Gradient_H: {}".format("good"), (420, 30), cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
                        poseStack_H = 0                    

                    # #몸이 기울었을때 -> 코와 목 사이 선의 기울기로 측정
                    if(abs(bodyGradient_V/UserFig.userFig.bodyGradient_V) < abs(UserFig.userLearn.bodyGradient_V*sensitivity)):
                        cv.putText(frame, "Gradient_V: {}".format("bad"), (420, 60), cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
                        poseStack_V += 1
                        if(poseStack_V == 200):
                            Upright_DB.Count_Habit("gradient")
                            path = "C:/imgfile/" + str("habit_") + Save_Cam() + ".png"
                            cv.imwrite(path, frame)
                            Upright_DB.insertBLOB(path)
                            messagebox.showinfo("경고", "자세가 기울었어요")
                            poseStack_H = 0
                            poseStack_V = 0
                    else:
                        cv.putText(frame, "Gradient_V: {}".format("good"), (420, 60), cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
                        poseStack_V = 0

                    # 거북목 체크 -> 코와 목 사이 선의 길이로 측정
                    if(BODY_LENGTH[17]/UserFig.userFig.bodyLength_R > UserFig.userLearn.bodyLength_R and BODY_LENGTH[18]/UserFig.userFig.bodyLength_L > UserFig.userLearn.bodyLength_L):
                        cv.putText(frame, "Turtle: {}".format("bad"), (420, 90), cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
                        poseStack_T += 1
                        if(poseStack_T == 100):
                            Upright_DB.Count_Habit("turtle")
                            path = "C:/imgfile/" + str("habit_") + Save_Cam() + ".png"
                            cv.imwrite(path, frame)
                            Upright_DB.insertBLOB(path)
                            messagebox.showinfo("경고", "거북목입니다!")
                            poseStack_T = 0
                    else:
                        cv.putText(frame, "Turtle: {}".format("good"), (420, 90), cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
                        poseStack_T = 0
                #----------------------------------------------------------------------------------------------------------------------------------

# 표정 인식-----------------------------------------------------------------------------------------------------------------------------------------------                
        for rect in rects:
            eyebrowshape = predictor(gray, rect)                    #눈썹
            eyebrowshape = face_utils.shape_to_np(eyebrowshape)

            eyeshape = predictor(gray, rect)
            eyeshape = face_utils.shape_to_np(eyeshape)

            mouthshape = predictor(gray, rect)
            mouthshape = face_utils.shape_to_np(mouthshape)

            noseshape = predictor(gray, rect)
            noseshape = face_utils.shape_to_np(noseshape)

            leftEye = eyeshape[lStart: lEnd]
            rightEye = eyeshape[rStart: rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            
            mouthOutLine = mouthshape[oStart:oEnd]
            mouthInnerLine = mouthshape[iStart:iEnd]
            mar = mouth_aspect_ratio(mouthOutLine, mouthInnerLine)

            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv.convexHull(leftEye)
            rightEyeHull = cv.convexHull(rightEye)
            cv.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)


            mouthOutHull = cv.convexHull(mouthOutLine)
            mouthInnerHull = cv.convexHull(mouthInnerLine)
            cv.drawContours(frame, [mouthOutHull], -1, (0, 255, 0), 1)
            cv.drawContours(frame, [mouthInnerHull], -1, (0, 255, 0), 1)

            # 처음 자세 보정
            if(clickNumber == 0):
                if(firstPose == 4):
                    Upright_DB.SaveFaceDB(ear, mar)
                    userFig = Upright_DB.Information_DB("first")                      #값을 보정하고 업데이트함
                    UserFig.userFig.Set_User(userFig[0], userFig[1], userFig[2], userFig[3], userFig[4], userFig[5])
                    path = "C:/imgfile/" + str("cor_") + Save_Cam() + ".png"
                    cv.imwrite(path, frame)
                    Upright_DB.UpdateBLOB(path)
                    messagebox.showinfo("완료", "보정이 완료되었습니다")
                    firstPose = 0
                    clickNumber = -1
                    break

            if(clickNumber == 2):
                if(badPose==6):
                    cv.putText(frame, "Status: {}".format("Eye Waiting..."), (0, 60), cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
                if(badPose == 7):
                    userE = round(ear/(UserFig.userFig.Ear),2)
                    messagebox.showinfo("완료", "눈 측정이 완료되었습니다")
                    messagebox.showinfo("알림", "입을 측정합니다\n" "입술을 깨물어주세요")
                    badPose = 8
                if(badPose==8):
                    cv.putText(frame, "Status: {}".format("Mouth Waiting..."), (0, 60), cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
                if(badPose == 9):
                    # userM = round(ear/(UserFig.userFig.Mar),2)
                    userM = round(mar,2)
                    Upright_DB.SaveInitLearningData(userH, userV, userR, userL, userE, userM, newORold)
                    UserFig.userLearn.Set_User(userH, userV, userR, userL, userE, userM)
                    messagebox.showinfo("완료", "입 측정이 완료되었습니다\n" "모든 측정이 완료되었습니다")
                    badPose = 0                     
                    clickNumber = -1
                    break

            #동작 감지 실행
            if(clickNumber == 1):   
                print(poseStack_E)      
                # 눈 찡그릴때                
                if(abs(ear/UserFig.userFig.Ear) < abs(UserFig.userLearn.Ear*(sensitivity))):
                    cv.putText(frame, "Eye: {}".format("bad"), (420, 120), cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
                    poseStack_E += 1
                    if(poseStack_E == 5):
                        Upright_DB.Count_Habit("eye")
                        path = "C:/imgfile/" + str("habit_") + Save_Cam() + ".png"
                        cv.imwrite(path, frame)
                        Upright_DB.insertBLOB(path)
                        messagebox.showinfo("경고","눈을 찡그리지 마세요")
                        poseStack_E = 0
                else:
                    cv.putText(frame, "Eye: {}".format("good"), (420, 120), cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
                    poseStack_E = 0

                #입술 깨물때
                if(abs(mar) < abs(UserFig.userLearn.Mar*(sensitivity))):
                    cv.putText(frame, "Mouth: {}".format("bad"), (420, 150), cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
                    poseStack_M += 1
                    if(poseStack_M == 5):
                        Upright_DB.Count_Habit("mouth")
                        path = "C:/imgfile/" + str("habit_") + Save_Cam() + ".png"
                        cv.imwrite(path, frame)
                        Upright_DB.insertBLOB(path)
                        messagebox.showinfo("경고","입술을 깨물지 마세요")
                        poseStack_M = 0
                else:
                    cv.putText(frame, "Mouth: {}".format("good"), (420, 150), cv.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
                    poseStack_M = 0
                        
            #------------------------------------------------------------------------------------------------------------------

        cv.imshow("original", frame)   # frame(카메라 영상)을 original 이라는 창에 띄워줌

        if cv.waitKey(1) & 0xFF == 27:  # 키보드의 esc를 누르면 창을 닫음
            break    
        if cv.getWindowProperty("original", cv.WND_PROP_VISIBLE) <1:  # X버튼을 누르면 창 닫기 -> 창이 닫혔는지 감지하면 종료, 다시 실행되는 것을 방지
            break

    cap.release()
    cv.destroyAllWindows()