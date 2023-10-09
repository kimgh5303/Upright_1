import sys
import User
import Upright_Pose
import Upright_DB
import webbrowser
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow
from tkinter import messagebox
from tkinter import Tk
import os

root=Tk()
root.withdraw()

def resource_path(relative_path): 
    """ Get absolute path to resource, works for dev and for PyInstaller """ 
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__))) 
    return os.path.join(base_path, relative_path)

form = resource_path('Login.ui')
form, base = uic.loadUiType(form)

form2 = resource_path('Main.ui')
form2, base2 = uic.loadUiType(form2)

form3 = resource_path('Join.ui')
form3, base3 = uic.loadUiType(form3)

class user_JoinWindow(base3, form3):
    
    def __init__(self):
        super(base3,self).__init__()
        #loadUi("Join.ui",self)
        self.setupUi(self)
        self.setFixedWidth(1280)
        self.setFixedHeight(900)
        self.pushButton_JoinJoin.clicked.connect(self.joinjoin)
        self.show()
    
    #회원가입
    def joinjoin(self):
        Upright_DB.Login_Save(self.lineEdit_id.text(), self.lineEdit_pw.text(), self.lineEdit_name.text(), self.lineEdit_birth.text(), self.lineEdit_addr.text())
        Upright_DB.First_Count_Habit(self.lineEdit_id.text())
        login=LoginWindow()
        widget.addWidget(login)
        widget.setCurrentIndex(widget.currentIndex()+1)

class MainWindow(base2, form2):
    def __init__(self):
        super(base2,self).__init__()
        #loadUi("Main.ui",self)
        self.setupUi(self)
        widget.setFixedWidth(750)
        widget.setFixedHeight(100)
        widget.move(300,100)
        
        MainWindow.Check_Sensitivity(self)                                  # 민감도 체크
        self.pushButton_Recode.clicked.connect(self.record)                 # 측정 버튼        
        self.pushButton_CamOn.clicked.connect(self.On)                      # 캠 온
        self.pushButton_correction.clicked.connect(self.correction)         # 보정
        self.pushButton_Measurement.clicked.connect(self.measurement)       # 측정
        self.pushButton_run.clicked.connect(self.run)                       # 실행
        self.pushButton_stop.clicked.connect(self.stop)                     # 실행 정지
        self.pushButton_Plus.clicked.connect(self.plus)                     # 민감도 +
        self.pushButton_Minus.clicked.connect(self.minus)                   # 민감도 -
        self.pushButton_Community.clicked.connect(self.community)               # 커뮤니티
        self.pushButton_Logout.clicked.connect(self.logout)
    
    # 사용자별 민감도 체크
    def Check_Sensitivity(self):
        if(User.user.stage==1):
            self.Sensitivity_label.setText("매우\n낮음")
            Upright_Pose.sensitivity = 0.8
        if(User.user.stage==2):
            self.Sensitivity_label.setText("낮음")
            Upright_Pose.sensitivity = 0.9
        if(User.user.stage==3):
            self.Sensitivity_label.setText("보통")
            Upright_Pose.sensitivity = 1
        if(User.user.stage==4):
            self.Sensitivity_label.setText("높음")
            Upright_Pose.sensitivity = 1.1
        if(User.user.stage==5):
            self.Sensitivity_label.setText("매우\n높음")
            Upright_Pose.sensitivity = 1.2

    def record(self):
        if(Upright_Pose.clickNumber==0):                                    # 클릭했을떄 다음 단계로 넘어가게 함, Upright_Pose 프로젝트로 번갈아 가면서 실행함
            if Upright_Pose.firstPose==2:Upright_Pose.firstPose=3
        if(Upright_Pose.clickNumber==2):
            if Upright_Pose.badPose==2:Upright_Pose.badPose=3
            if Upright_Pose.badPose==4:Upright_Pose.badPose=5
            if Upright_Pose.badPose==6:Upright_Pose.badPose=7
            if Upright_Pose.badPose==8:Upright_Pose.badPose=9

    def On(self):
        Upright_Pose.running = True
        Upright_Pose.camStart()
        
    def correction(self):
        try:
            if(Upright_Pose.clickNumber == -1):
                Upright_Pose.clickNumber = 0
        except AttributeError:
            messagebox.showinfo("알림", "카메라를 켜주세요")

    def measurement(self):
        try:
            if(Upright_Pose.clickNumber == -1):
                Upright_Pose.clickNumber = 2
        except AttributeError:
            messagebox.showinfo("알림", "카메라를 켜주세요")
    def run(self):
        Upright_Pose.clickNumber = 1

    def stop(self):
        Upright_Pose.clickNumber = -1
        Upright_Pose.firstPose = 0
        Upright_Pose.badPose = 0
        Upright_Pose.poseStack_H = Upright_Pose.poseStack_V = Upright_Pose.poseStack_T = Upright_Pose.poseStack_E = Upright_Pose.poseStack_M = 0

    def plus(self):
        if(User.user.stage<5):
            User.user.stage+=1
            Upright_DB.Change_Stage()
            MainWindow.Check_Sensitivity(self)

    def minus(self):
        if(User.user.stage>1):
            User.user.stage-=1
            Upright_DB.Change_Stage()
            MainWindow.Check_Sensitivity(self)

    def community(self):
        webbrowser.open(url='http://ec2-18-188-52-122.us-east-2.compute.amazonaws.com/')

    def logout(self):
        exit()

class LoginWindow(base, form):

    def __init__(self):
        super(base, self).__init__()
        #loadUi("Login.ui",self)'
        self.setupUi(self)
        self.setFixedWidth(1000)
        self.setFixedHeight(600)
        self.pushButton_Login.clicked.connect(self.switch)
        self.pushButton_Join.clicked.connect(self.join)
    
    def join(self):
        join=user_JoinWindow()
        widget.addWidget(join)
        widget.setCurrentIndex(widget.currentIndex()+1)
    def switch(self):
        correct=Upright_DB.LoginORJoin_Load(self.lineEdit_ID.text(), self.lineEdit_Password.text(), "Login")
        if correct==True:
            Upright_DB.LoginWho()
            main=MainWindow()
            widget.addWidget(main)
            widget.setCurrentIndex(widget.currentIndex()+1)
        else:
            messagebox.showinfo("에러", "일치하지 않습니다!")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = LoginWindow()
    widget=QtWidgets.QStackedWidget()
    widget.setFixedWidth(1000)
    widget.setFixedHeight(600)
    widget.addWidget(win)
    widget.show()
    app.exec_()

    '''
'''