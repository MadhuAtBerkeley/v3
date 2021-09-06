import paho.mqtt.client as mqtt
import cv2
import numpy as np
from os import system
import sys

# define local client
CLOUD_MQTT_HOST="cloud_mqtt_broker"
CLOUD_MQTT_PORT=1883
CLOUD_MQTT_TOPIC="face_detect"

def on_connect_cloud(client, userdata, flags, rc):
        print("connected to cloud broker with rc: " + str(rc))
        client.subscribe(CLOUD_MQTT_TOPIC)
	

def on_message_cloud(client, userdata, msg):
  try:
    #i = int(msg.payload[0])   # get message number
    jpg_msg = msg.payload
    print("Received message with size" {}".format(len(jpg_msg)))	    
    jpg_img = cv2.imdecode(np.frombuffer(jpg_msg, dtype='uint8'),cv2.IMREAD_COLOR)
    cv2.imwrite('/home/s3fs_data/face.jpg', jpg_img)
   
  except:
    print("Unexpected error:", sys.exc_info()[0])

#start cloud client
cloud_mqttclient = mqtt.Client()
cloud_mqttclient.on_connect = on_connect_cloud
cloud_mqttclient.connect(CLOUD_MQTT_HOST, CLOUD_MQTT_PORT, 60)
cloud_mqttclient.on_message = on_message_cloud

cloud_mqttclient.loop_forever()
