
import cv2
import os
import time
from time import sleep
import numpy as np
import pytesseract
import threading
from actions import doAction
import nxbt
import copy
import torch
from predictor import divide, predict
#pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
resetMacro = """
L R A 1s
"""
def grayscale(img):
    original = np.array(img) 
    original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    gray_scale = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    return gray_scale
    
def checkText(img):
    text = pytesseract.image_to_string(img)
    #print(text)
    if 'VAMPIRE' in text:
        return True
    return False
state = 0
ministate = 0
image =np.random.random((5, 5))
distance = 1
def getState():
    global state
    global ministate
    global image
    cap = cv2.VideoCapture(0)
    # The device number might be 0 or 1 depending on the device and the webcam
    while(True):
        ret, frame = cap.read()
        cv2.imshow('Switch', frame)
        frame2 =copy.deepcopy(frame)
        image = copy.deepcopy(frame)
        state = np.moveaxis(frame, -1, 0).astype(np.double) # put channels in first dimension
        state = torch.tensor(state.copy(), dtype=torch.double)
        
        
        
        frame2 = cv2.resize(frame2, (100, 100), interpolation=cv2.INTER_CUBIC)
        frame2 = np.moveaxis(frame2, -1, 0).astype(np.double) # put channels in first dimension     
        ministate = torch.tensor(frame2.copy(), dtype=torch.double)
        #ministate = torch.unsqueeze(ministate, 0)
        
        cv2.waitKey(1)
def predictDistance():
    global distance
    global image
    while True:
        imgs, w, h = divide(image, 5)
        predictions = []
        enemyx = 0
        enemyy = 0
        playerx = 0 
        playery = 0
        
        for i in range(len(imgs)):
            img, x, y = imgs[i]
            pred = predict(img)
            if pred == 1:
                enemyx= x
                enemyy= y
            elif pred== 2:
                playerx= x
                playery = y
        distance =  abs(enemyx - playerx) + abs(enemyy - playery)
t = threading.Thread(target=getState)
t.start()
sleep(1)
i = threading.Thread(target=predictDistance)
i.start()
print(image.shape)
class env():
    def __init__(self, inverted = False, controller_index = 0):
        self.draw_window = True
        
        self.inverted = inverted
        self.curHp = 0
        self.curEnemyHp = 0   
        self.last_hp = 0
        self.last_enemy_hp = 0
        self.gotHit = 0
        
        self.frame = 0
        self.hitEnemy = 0
        input("Press Enter when ready to connect controller")
        self.nx = nxbt.Nxbt()
        self.controller_index = self.nx.create_controller(nxbt.PRO_CONTROLLER)
        print("Controller index: " + str(self.controller_index))
        self.nx.wait_for_connection(self.controller_index)
        print("Connecting Controller Please Wait")
        print(f"Shape of Input {state.shape}, {ministate.shape}")
        print("Connected Controller")
        input("Press Enter to start when ready")
        print("Started")
    def setHps(self):
        self.curHp = state[1, 155, 435]
        self.curEnemyHp = state[1, 295, 435]
        if self.inverted:
            self.curHp = state[1, 295, 435]
            self.curEnemyHp = state[1, 155, 435]
            
    def getReward(self, action):
        reward = -1
        self.setHps()
        if(self.curHp != self.last_hp): # if hp changed
            self.gotHit = 1
            #reward -=20
       
        
        
        hit = 0
        if(self.curEnemyHp != self.last_enemy_hp): # if enemy hp changed
            self.hitEnemy = 1
            
        return reward
    def step(self, action):
        last_distance = distance
        done = False
        if self.frame >= 1000:
            self.frame = 0
            done = True
        self.hitEnemy = 0
        self.gothit = 0
        reward = 0
        
        self.last_hp = state[1, 155, 435]
        self.last_enemy_hp = state[1, 295, 435]
        lag, actionReward = doAction(self.nx, self.controller_index, action)
        
        while(lag > 0 or self.gotHit!=1):
            lag -= 1
            reward += self.getReward(action)
            self.frame += 1
            time.sleep(0.01)
        if(distance < last_distance & action <= 4):
            reward +=5
        elif(distance > last_distance & action > 4):
            reward -=10
        if self.hitEnemy == 1:
            reward += actionReward
        else:
            reward -= actionReward
        if self.gotHit:
            reward -=20
        next_state = ministate

        info = []
        return next_state, reward, done, info, distance
    def reset(self):
        self.gothit = 0
        self.hitEnemy =0
        time.sleep(.5)
        self.nx.macro(self.controller_index, resetMacro)        
        next_state = ministate
        return next_state
        