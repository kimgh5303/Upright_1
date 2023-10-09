from io import StringIO
import pymysql
import Ui
import User
import UserFig
from mysql.connector import connection
import mysql.connector
import sys
import base64
from PIL import Image
from tkinter import messagebox
from tkinter import Tk

root=Tk()
root.withdraw()

firstData = []
secondData = []

#로그인 관련------------------------------------------------------------------------------------------------------------------------------------------------
#회원가입
def Login_Save(id,pw,name,birth,addr):
    upRight_db = pymysql.connect(host="210.125.31.247",user="test2",password="s1234",db="upright", charset="utf8")
    cursor = upRight_db.cursor()
    sql = "INSERT `upright`.`user_table` SET `id` = %s, `pw` = %s, `name` = %s, `birth` = %s , `addr` = %s, `stage` = %s;"
    val = (id,pw,name,birth,addr, 3)
    cursor.execute(sql, val)
    upRight_db.commit()
    #messagebox.showinfo("회원가입", "정상적으로 가입되었습니다!")
    return

#로그인
def LoginORJoin_Load(id,pw,LoginORJoin):
    upRight_db = pymysql.connect(host="210.125.31.247",user="test2",password="s1234",db="upright", charset="utf8")
    cursor = upRight_db.cursor()
    sql = "SELECT id, pw, name, stage FROM upright.user_table WHERE id='"+id+"';"
    cursor.execute(sql)
    row = cursor.fetchone()

    if(row is not None):
        User.user.Set_User(row[0], row[1], row[2], row[3])
        if(LoginORJoin == "Join"):
            return True
    else: 
        return False

    if(LoginORJoin == "Login"):
        if pw==User.user.pw:
            return True
        else:
            return False
    upRight_db.close()

#로그인 누가 했는지
def LoginWho():
    upRight_db = pymysql.connect(host="210.125.31.247",user="test2",password="s1234",db="upright", charset="utf8")
    cursor = upRight_db.cursor()
    sql = "UPDATE `upright`.`now_id` SET `id` = %s;"
    val = (User.user.id)
    cursor.execute(sql, val)
    upRight_db.commit()

#유저 민감도 조정
def Change_Stage():
    upRight_db = pymysql.connect(host="210.125.31.247",user="test2",password="s1234",db="upright", charset="utf8")
    cursor = upRight_db.cursor()
    sql = "UPDATE `upright`.`user_table` SET `stage` = %s WHERE (`id` = %s);"
    val = (User.user.stage, User.user.id)
    cursor.execute(sql, val)
    upRight_db.commit()
    return

#---------------------------------------------------------------------------------------------------------------------------------------------------

#보정------------------------------------------------------------------------------------------------------------------------------------------------
#몸 동작 관련 수치 저장
def SaveBodyDB(first_bodyGradient_H, first_bodyGradient_V, first_bodyLength_R, first_bodyLength_L, newORold):
    upRight_db = pymysql.connect(host="210.125.31.247",user="test2",password="s1234",db="upright", charset="utf8")
    cursor = upRight_db.cursor()
    #기존 유저
    if(newORold == "old"):
        sql = "UPDATE `upright`.`user_information` SET `bodyGradient_H` = %s, `bodyGradient_V` = %s, `bodyLength_R` = %s, `bodyLength_L` = %s WHERE (`id` = %s);"
        val = (first_bodyGradient_H, first_bodyGradient_V, first_bodyLength_R, first_bodyLength_L, User.user.id)
    #신규 유저
    elif(newORold == "new"):
        sql = "INSERT `upright`.`user_information` SET `id` = %s, `bodyGradient_H` = %s, `bodyGradient_V` = %s, `bodyLength_R` = %s, `bodyLength_L` = %s;" 
        val = (User.user.id, first_bodyGradient_H, first_bodyGradient_V,  first_bodyLength_R, first_bodyLength_L)

    cursor.execute(sql, val)
    upRight_db.commit()
    return

#표정 관련 수치 저장
def SaveFaceDB(firstEar, firstMar):
    upRight_db = pymysql.connect(host="210.125.31.247",user="test2",password="s1234",db="upright", charset="utf8")
    cursor = upRight_db.cursor()
    sql = "UPDATE `upright`.`user_information` SET `firstEar` = %s, `firstMar` = %s WHERE (`id` = %s);"
    val = (firstEar, firstMar, User.user.id)
    cursor.execute(sql, val)
    upRight_db.commit()
    return

#관련 수치들을 불러옴
def Information_DB(order):
    firstData.clear()
    secondData.clear()
    upRight_db = pymysql.connect(host="210.125.31.247",user="test2",password="s1234",db="upright", charset="utf8")
    cursor = upRight_db.cursor()
    if(order=="first"):
        sql = "SELECT bodyGradient_H, bodyGradient_V, bodyLength_R, bodyLength_L, firstEar, firstMar FROM upright.user_information WHERE id='"+User.user.id+"';"
    elif(order=="second"):
        sql = "SELECT bodyGradient_H, bodyGradient_V, bodyLength_R, bodyLength_L, eye, mouth FROM upright.learningdata WHERE id='"+User.user.id+"';"
    cursor.execute(sql)
    row = cursor.fetchone()
    
    if(row is not None):
        for r in row:
            if(order=="first"):
                firstData.append(r)
            elif(order=="second"):
                secondData.append(r)
    upRight_db.close()
    if(order=="first"):
        return firstData
    elif(order=="second"):
        return secondData
#---------------------------------------------------------------------------------------------------------------------------------------------------

#실행------------------------------------------------------------------------------------------------------------------------------------------------
def SaveInitLearningData(userH, userV, userR, userL, userE, userM, newORold):
    upRight_db = pymysql.connect(host="210.125.31.247",user="test2",password="s1234",db="upright", charset="utf8")
    cursor = upRight_db.cursor()
    if(newORold == "old"):
        sql = "UPDATE `upright`.`learningdata` SET `bodyGradient_H` = %s, `bodyGradient_V` = %s, `bodyLength_R` = %s, `bodyLength_L` = %s, `eye` = %s, `mouth` = %s WHERE (`id` = %s);"
        val = (userH, userV, userR, userL, userE, userM, User.user.id)
    #신규 유저
    elif(newORold == "new"):
        sql = "INSERT `upright`.`learningdata` SET `id` = %s, `bodyGradient_H` = %s, `bodyGradient_V` = %s, `bodyLength_R` = %s, `bodyLength_L` = %s, `eye` = %s, `mouth` = %s;" 
        val = (User.user.id, userH, userV, userR, userL, userE, userM)
    cursor.execute(sql, val)
    upRight_db.commit()

#초기 통계 집어넣기
def First_Count_Habit(id):
    upRight_db = pymysql.connect(host="210.125.31.247",user="test2",password="s1234",db="upright", charset="utf8")
    cursor = upRight_db.cursor()
    sql = "INSERT `upright`.`habit_count` SET `id` = %s, `gradient` = 0, `turtle` = 0, `eye` = 0, `mouth` = 0;" 
    val = (id)
    cursor.execute(sql, val)
    upRight_db.commit()

#통계 집어넣기
def Count_Habit(habit):
    upRight_db = pymysql.connect(host="210.125.31.247",user="test2",password="s1234",db="upright", charset="utf8")
    cursor = upRight_db.cursor()
    if(habit=="gradient"):sql = "UPDATE `upright`.`habit_count` SET `gradient`=`gradient`+1 WHERE (`id` = %s);"
    elif(habit=="turtle"):sql = "UPDATE `upright`.`habit_count` SET `turtle`=`turtle`+1 WHERE (`id` = %s);"
    elif(habit=="eye"):sql = "UPDATE `upright`.`habit_count` SET `eye`=`eye`+1 WHERE (`id` = %s);"
    elif(habit=="mouth"):sql = "UPDATE `upright`.`habit_count` SET `mouth`=`mouth`+1 WHERE (`id` = %s);"
    val = (User.user.id)
    cursor.execute(sql, val)
    upRight_db.commit()
#---------------------------------------------------------------------------------------------------------------------------------------------------

#이미지 저장-----------------------------------------------------------------------------------------------------------------------------------------
connection2 = pymysql.NULL

def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        binaryData = file.read()
    return binaryData

#보정 이미지 넣기
def UpdateBLOB(photo):
    global connection2
    try:
        connection2 = pymysql.connect(host="210.125.31.247",user="test2",password="s1234",db="upright", charset="utf8")

        cursor = connection2.cursor()
        # sql_insert_blob_query = """ INSERT INTO user_information (img) VALUES (%s)"""
        sql_insert_blob_query = """ UPDATE user_information SET img=%s WHERE id=%s"""

        empPicture = convertToBinaryData(photo)

        # Convert data into tuple format
        insert_blob_tuple = (empPicture, User.user.id)
        result = cursor.execute(sql_insert_blob_query, insert_blob_tuple)
        connection2.commit()

    except mysql.connector.Error as error:
        print("Failed inserting BLOB data into MySQL table {}".format(error))

#습관 자세 이미지 넣기
def insertBLOB(photo):
    global connection2
    try:
        connection2 = pymysql.connect(host="210.125.31.247",user="test2",password="s1234",db="upright", charset="utf8")

        cursor = connection2.cursor()
        sql_insert_blob_query = """ INSERT INTO imgList (id, img) VALUES (%s, %s)"""

        empPicture = convertToBinaryData(photo)

        # Convert data into tuple format
        insert_blob_tuple = (User.user.id, empPicture)
        result = cursor.execute(sql_insert_blob_query, insert_blob_tuple)
        connection2.commit()

    except mysql.connector.Error as error:
        print("Failed inserting BLOB data into MySQL table {}".format(error))