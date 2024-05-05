import RPi.GPIO as GPIO 

import time 

GPIO.setmode(GPIO.BCM) 

GPIO_laser = 21
GPIO.setup(GPIO_laser, GPIO.OUT)


GPIO.output(GPIO_laser, GPIO.HIGH)
time.sleep(5)
GPIO.output(GPIO_laser, GPIO.LOW)

GPIO.cleanup()
