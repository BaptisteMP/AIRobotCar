# -*- coding: utf-8 -*-
"""
Created on Wed May  1 10:36:20 2019

@author: Baptiste
"""

import RPi.GPIO as GPIO
from time import sleep
 
GPIO.setmode(GPIO.BOARD)
 
class Motor:
 
    def __init__(self, pinForward, pinBackward, pinControl):
        """ Initialize the motor with its control pins and start pulse-width
             modulation """
 
        self.pinForward = pinForward
        self.pinBackward = pinBackward
        self.pinControl = pinControl
        GPIO.setup(self.pinForward, GPIO.OUT)
        GPIO.setup(self.pinBackward, GPIO.OUT)
        GPIO.setup(self.pinControl, GPIO.OUT)
        self.pwm_forward = GPIO.PWM(self.pinForward, 100)
        self.pwm_backward = GPIO.PWM(self.pinBackward, 100)
        self.pwm_forward.start(0)
        self.pwm_backward.start(0)
        GPIO.output(self.pinControl,GPIO.HIGH) 
 
    def forward(self, speed):
        """ pinForward is the forward Pin, so we change its duty
             cycle according to speed. """
        self.pwm_backward.ChangeDutyCycle(0)
        self.pwm_forward.ChangeDutyCycle(speed)    
 
    def backward(self, speed):
        """ pinBackward is the forward Pin, so we change its duty
             cycle according to speed. """
 
        self.pwm_forward.ChangeDutyCycle(0)
        self.pwm_backward.ChangeDutyCycle(speed)
 
    def stop(self):
        """ Set the duty cycle of both control pins to zero to stop the motor. """
 
        self.pwm_forward.ChangeDutyCycle(0)
        self.pwm_backward.ChangeDutyCycle(0)
 
motor1 = Motor(11, 15, 7)
motor2 = Motor(16, 18, 8)

def race():
    
    speeds = [0, 0] #speedmot1, speedmot2
    direct = 0 #forwardmot1= 1, backwardmot1 = -1, same for mot2
     
    while(True):
        
        print('z: forward, s: backward, q: turn left, d: turn right, 0: stop, 1: slow speed, 2: normal speed, 3:fast speed, stop: end the program', end='\r')

        a = input()

        if a == '1' or a == '2' or a == '3':
            s = 20*int(a)
            speeds = [s, s]
            if direct > -1:
                motor1.forward(s)
                motor2.forward(s)
                direct = 1
            else: 
                motor1.backward(s)
                motor2.backward(s)
                   
        elif a == '0':
            motor1.stop()
            motor2.stop()
            direct = 0
            
        elif a == 'z':
             motor1.forward(speeds[0])
             motor2.forward(speeds[1])
             direct = 1
        
        elif a == 's':
             motor1.backward(speeds[0])
             motor2.backward(speeds[1])
             direct = -1
        
        elif a == 'q': #turn left
            if direct == 1:
                motor1.forward(speeds[0])
                motor2.forward(speeds[1]-5)
                speeds[1] -= 10
            elif direct == -1:
                motor1.backward(speeds[0])
                motor2.backward(speeds[1]-5)
                speeds[1] -= 10
                
        elif a == 'd': #turn right
            if direct == 1:
                motor2.forward(speeds[1])
                motor1.forward(speeds[0]-10)
                speeds[0] -= 10
            elif direct == -1:
                motor2.backward(speeds[1])
                motor1.backward(speeds[0]-10)
                speeds[0] -= 10
        
        elif a == 'stop':
            motor1.stop()
            motor2.stop()
            break

        sleep(0.2)

race()
print('race ended')

# Running both
motor1.forward(20)
motor2.backward(70)
sleep(5)
motor1.forward(90)
sleep(5)
motor1.stop()
motor2.stop()
 
 
GPIO.cleanup()