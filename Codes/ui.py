# Copyright (c) Acconeer AB, 2022
# All rights reserved
import pickle
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
import sys
sys.path.append('/home/pi/lcd')
import lcd
import time
import numpy as np
import pyrebase
import datetime
timestamp = datetime.datetime.now()
print(timestamp.strftime("%m/%d/%Y, %H:%M:%S"))
count = 0
# import pyqtgraph as pg
# from firebase import firebase

import acconeer.exptool as et
config={
   "apiKey": "AIzaSyCSI1XHYpvzVHFawjEKMlJnlrjJTnN6YdQ",
   "authDomain": "vital-e5b32.firebaseapp.com",
   "databaseURL": "https://vital-e5b32-default-rtdb.firebaseio.com/",
   "storageBucket": "vital-e5b32.appspot.com"

}
firebase = pyrebase.initialize_app(config)
db = firebase.database()
heart_rate_values =[]
breath_rate_values =[]
# firebase=firebase.FirebaseApplication('https://xm112xb112-default-rtdb.asia-southeast1.firebasedatabase.app/',None)

# setting up ultrasonic sensor and laser
GPIO_laser = 21
GPIO.setup(GPIO_laser, GPIO.OUT)
GPIO_TRIG = 23
GPIO_ECHO = 24
GPIO.setup(GPIO_TRIG, GPIO.OUT) 
GPIO.setup(GPIO_ECHO, GPIO.IN)
def obj():
    per = False
    GPIO.output(GPIO_TRIG, GPIO.LOW) 
    time.sleep(1) 
    GPIO.output(GPIO_TRIG, GPIO.HIGH) 
    time.sleep(0.00001) 
    GPIO.output(GPIO_TRIG, GPIO.LOW) 

    while GPIO.input(GPIO_ECHO)==0: 
        start_time = time.time() 

    while GPIO.input(GPIO_ECHO)==1: 
        Bounce_back_time = time.time() 

    pulse_duration = Bounce_back_time - start_time 
    distance = round((pulse_duration * 17150)-5.5, 2) 
    print (f"Distance: {distance} cm") 
    if(distance<40 or distance>100):
        lcd.lcd_byte(lcd.LCD_LINE_1, lcd.LCD_CMD)
        lcd.lcd_string("Object not", 2)
        lcd.lcd_byte(lcd.LCD_LINE_2, lcd.LCD_CMD)
        lcd.lcd_string("Detected", 2)
        time.sleep(1)
        distance = 0
        per=False
    else:
        distance = 0
        per=True
    return per

GPIO.output(GPIO_laser, GPIO.LOW)
GPIO.output(GPIO_laser, GPIO.HIGH)


lcd.lcd_init()
lcd.lcd_byte(lcd.LCD_LINE_1, lcd.LCD_CMD)
lcd.lcd_string("Patient Name:", 2)
Name=input("Enter the patient name: ")
lcd.lcd_byte(lcd.LCD_LINE_2, lcd.LCD_CMD)
lcd.lcd_string(Name, 2)
time.sleep(2)
lcd.lcd_byte(lcd.LCD_LINE_2, lcd.LCD_CMD)
lcd.lcd_string("        ", 2)
lcd.lcd_byte(lcd.LCD_LINE_1, lcd.LCD_CMD)
lcd.lcd_string("Patient Age:", 2)
Age = input("Enter the patient Age: ")
lcd.lcd_byte(lcd.LCD_LINE_2, lcd.LCD_CMD)
lcd.lcd_string(Age, 2)
time.sleep(2)
lcd.lcd_byte(lcd.LCD_LINE_2, lcd.LCD_CMD)
lcd.lcd_string("        ", 2)
lcd.lcd_byte(lcd.LCD_LINE_1, lcd.LCD_CMD)
lcd.lcd_string("Patient ID:", 2)
ID=input("Enter the patient ID: ")
lcd.lcd_byte(lcd.LCD_LINE_2, lcd.LCD_CMD)
lcd.lcd_string(ID, 2)
time.sleep(2)
lcd.lcd_byte(lcd.LCD_LINE_2, lcd.LCD_CMD)
lcd.lcd_string("        ", 2)
lcd.lcd_byte(lcd.LCD_LINE_1, lcd.LCD_CMD)
lcd.lcd_string("Patient Gender:", 2)
Gender=input("Enter the patient gender: ")
lcd.lcd_byte(lcd.LCD_LINE_2, lcd.LCD_CMD)
lcd.lcd_string(Gender, 2)
time.sleep(2)
lcd.lcd_byte(lcd.LCD_LINE_1, lcd.LCD_CMD)
lcd.lcd_string("Initiated   ", 2)
lcd.lcd_byte(lcd.LCD_LINE_2, lcd.LCD_CMD)
lcd.lcd_string(" ", 2)

# object detection
while (1):
    per=obj()
    if(per):
        per=obj()
        if(per):
            per=obj()
            if(per):
                break

lcd.lcd_byte(lcd.LCD_LINE_1, lcd.LCD_CMD)
lcd.lcd_string("Initiated   ", 2)
lcd.lcd_byte(lcd.LCD_LINE_2, lcd.LCD_CMD)
lcd.lcd_string(" ", 2)

GPIO.output(GPIO_laser, GPIO.LOW)


class PGUpdater:
    def __init__(self, sensor_config, processing_config, session_info):
        self.config = sensor_config

    # def setup(self, win):
    #    print("no window")

    def update(self, data):
        
         global Name
         global Age
         global ID
         global Gender
         global count
         if data["init_progress"] is not None:
            print("Initiating: {} %".format(data["init_progress"]))
         else:
            snr = data["snr"]
            if snr == 0:
                s = "SNR: N/A | {:.0f} dB".format(10 * np.log10(data["lambda_p"]))
            else:
                fmt = "SNR: {:.0f} | {:.0f} dB"
                s = fmt.format(10 * np.log10(snr), 10 * np.log10(data["lambda_p"]))
            f_est = data["f_dft_est"]
            f_estbr=data["f_estBR"]
            # print(f_est,"inside ui")
            global heart_rate_values 
            global breath_rate_values 
            lcd.lcd_byte(lcd.LCD_LINE_1, lcd.LCD_CMD)
            lcd.lcd_string("Estimating ...", 2)
            lcd.lcd_byte(lcd.LCD_LINE_2, lcd.LCD_CMD)
            lcd.lcd_string("Please Wait", 2)

# Assuming this code is inside a loop or a function
# ...

# Update heart rate and breath rate values
            if f_estbr > 0 and f_estbr<0.8 and f_est > 1.1:
                while (1):
                    per=obj()
                    if(per):
                        break
                heart_rate_values.append(int(f_est * 60))
                breath_rate_values.append(int(f_estbr * 60))
                print("Heart_rate: ",heart_rate_values)
                print("Breath rate: ",breath_rate_values)
                if len(heart_rate_values) >= 3 and len(breath_rate_values)>=3:
    # Calculate average heart rate and breath rate
                    avg_heart_rate = int(sum(heart_rate_values[-3:]) / 3)
                    avg_breath_rate = int(sum(breath_rate_values[-3:]) / 3)

    # Print average values
                    s1 = "Breath rate: {:.0f} ".format(avg_breath_rate)
                    print(s1)
                    s = "Heart rate: {:.0f} ".format(avg_heart_rate)
                    print(s)
                    lcd.lcd_byte(lcd.LCD_LINE_1, lcd.LCD_CMD)
                    lcd.lcd_string(s1, 1)
                    lcd.lcd_byte(lcd.LCD_LINE_2, lcd.LCD_CMD)
                    lcd.lcd_string(s,1)
                    time.sleep(2)
                    count=count+1
                    timestamp = datetime.datetime.now()
                    t=timestamp.strftime("%m/%d/%Y, %H:%M:%S")
                    data1={
                    "Name": Name,
                    "Age": Age,
                    "Gender":Gender,
                    "ID": ID,
                    "heart_rate": f_est*60,
                    "breath_rate": int(f_estbr*60),
                    "Timestamp":t,
                    
                    }
                    time.sleep(2)
                    lcd.lcd_byte(lcd.LCD_LINE_1, lcd.LCD_CMD)
                    lcd.lcd_string("Estimating....", 2)
                    lcd.lcd_byte(lcd.LCD_LINE_2, lcd.LCD_CMD)
                    lcd.lcd_string("Please wait", 2)
                    try:                       
                        db.child("mlx90614").child("1-set").set(data1)
                        db.child("mlx90614").child("2-push").push(data1)
                    except:
                        print("Unable to sent to cloud")
                    
                    heart_rate_values = heart_rate_values[-3:]
                    breath_rate_values = breath_rate_values[-3:]
                    if count>5:
                        GPIO.output(GPIO_laser, GPIO.LOW)
                        GPIO.cleanup()
                        exit()
