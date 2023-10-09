
class UserFig:
    bodyGradient_H: float
    bodyGradient_V: float
    bodyLength_R: int
    bodyLength_L: int
    Ear: float
    Mar: float

    def Set_User(self, bodyGradient_H: float, bodyGradient_V: float, bodyLength_R: int, bodyLength_L: int, Ear: float, Mar: float):
        self.bodyGradient_H = bodyGradient_H
        self.bodyGradient_V = bodyGradient_V
        self.bodyLength_R = bodyLength_R
        self.bodyLength_L = bodyLength_L
        self.Ear = Ear
        self.Mar = Mar

    def Set_User_Body(self, bodyGradient_H: float, bodyGradient_V: float, bodyLength_R:int, bodyLength_L: int):
        self.bodyGradient_H = bodyGradient_H
        self.bodyGradient_V = bodyGradient_V
        self.bodyLength_R = bodyLength_R
        self.bodyLength_L = bodyLength_L
        
    def Set_User_Face(self, Ear: float, Mar: float):
        self.Ear = Ear
        self.Mar = Mar
        
userFig = UserFig()     # 보정값을 위한 유저 객체
userLearn = UserFig()   # 학습값을 위한 유저 객체