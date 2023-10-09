
class UserInfo:
    id: str
    pw: str
    name: str
    stage: int

    def Set_User(self, id: str, pw: str, name: str, stage: str):
        self.id = id
        self.pw = pw
        self.name = name
        self.stage = stage
        
user = UserInfo()
