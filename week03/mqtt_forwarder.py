import paho.mqtt.client as mqtt
import time
import sys
  
LOCAL_MQTT_HOST="nx_mqtt_broker"
LOCAL_MQTT_PORT=1883
LOCAL_MQTT_TOPIC="face_detect"

REMOTE_MQTT_HOST="54.186.188.18"
REMOTE_MQTT_PORT=1883
REMOTE_MQTT_TOPIC="face_detect"


def on_connect_local(client, userdata, flags, rc):
        print("connected to local broker with rc: " + str(rc))
        client.subscribe(LOCAL_MQTT_TOPIC)
	

def on_message_local(client, userdata, msg):
  try:
    print('Message received with len:{}'.format(len(msg.payload)))
    # if we wanted to re-publish this message, something like this should work
    remote_mqttclient.publish(REMOTE_MQTT_TOPIC, payload=msg.payload, qos=0, retain=False)
    #print('Message forwarded')
  except:
    print("Unexpected error:", sys.exc_info()[0])


#initialize remote client
remote_mqttclient = mqtt.Client()
remote_mqttclient.connect(REMOTE_MQTT_HOST, REMOTE_MQTT_PORT, 60)

#initialize local client
local_mqttclient = mqtt.Client()
local_mqttclient.on_connect = on_connect_local
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
local_mqttclient.on_message = on_message_local

#wait in forever loop
local_mqttclient.loop_forever()
