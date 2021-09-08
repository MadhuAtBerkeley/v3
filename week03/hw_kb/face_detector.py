import paho.mqtt.client as mqtt
import time
import numpy as np
import cv2
  

LOCAL_MQTT_HOST="mosquitto-service"
LOCAL_MQTT_PORT= 1883  
LOCAL_MQTT_TOPIC="loc_0/face_detect"

def on_connect_local(client, userdata, flags, rc):
        print("connected to local broker with rc: " + str(rc))

local_mqttclient = mqtt.Client()
local_mqttclient.on_connect = on_connect_local
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)



face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

i = 0
while(True):
 
    ret, frame = cap.read()

    time.sleep(1)
    # if frame is captured correctly
    if ret == True:
        print("captured image")
        # convert image to gray scale
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #detect faces
        faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)

        #for every detected face draw rectangular bounding box and send
        for (x, y, w, h) in faces:
            # create a rectangle around the face
            face = cv2.rectangle(gray_img, (x,y), (x+w,y+h), (255,0,0), 2)

            #convert to msg bytes
            jpg_msg = cv2.imencode('.jpg', face)[1].tobytes()
            
            #add message/image number
            msg = bytes([i])+jpg_msg
            
            # forward to mqtt broker on jetson
            print("publishing message")
         
            local_mqttclient.publish(LOCAL_MQTT_TOPIC, payload=msg, qos=1, retain=False)
            
            #Save 5 images locally
            i += 1
            if (i == 5): i = 0
            #cv2.imwrite('/face'+str(i)+'.png', face)
            time.sleep(5)


    #(cv2.waitKey(1) & 0xFF) == ord('q'):
    #    break
    # Displays the window infinitely
   

local_mqttclient.loop_stop()
local_mqttclient.disconnect()
cap.release()
cv.destroyAllWindows()


