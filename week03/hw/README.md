# Homework 3

## On Jetson NX

1. Create a bridge network
`docker network create --driver bridge hw03`

2. MQTT broker

    * Build the image

        `sudo docker build -t nx_mqtt_broker -f dockerfile_mqtt_broker .`

    * Spin up container and establish broker

        `sudo docker run --rm --name nx_mqtt_broker --network hw03 -p 1883:1883 -ti mqtt_broker /usr/sbin/mosquitto`

3. MQTT forwarder

    This spins up the forwarder that connects with both broker. So the remote broker in the cloud must be started before running forwarder.

    * Build the image

        `sudo docker build -t nx_mqtt_forwarder -f dockerfile_mqtt_forwarder .`

    * Spin up container and run `nx_forwarder.py`

        `sudo docker run --rm --name nx_mqtt_forwarder --network hw03 -v ~/work/W251/v3/week03/hw/:/home/ -ti nx_mqtt_forwarder /bin/sh /home/mqtt_forwarder.sh`

4. OpenCV face detector

    * Build the image

        `sudo docker build -t nx_face_detector -f dockerfile_face_detector .`

    * Spin up the container and run `nx_face_detector.py`

        `sudo docker run --rm --privileged -e DISPLAY --name nx_face_detector --network hw03 -v ~/work/W251/v3/week03/hw/:/home/ -ti nx_face_detector /bin/bash /home/nx_face_detector.sh`

   A timeout of 5 seconds is added with counters to keep track of faces/pictures. This is 5 second is chosen based on the use case (where prisoners are let-in one person at a time with gap of 10 seconds). Tests have been done with this restriction removed and it works fine. 

## On AWS

1. Create a bridge network
`sudo docker network create --driver bridge hw03`

2. Cloud MQTT broker

    * Build the image

        `sudo docker build -t cloud_mqtt_broker -f dockerfile_cloud_broker .`

    * Spin up container and establish broker

        `sudo docker run --rm --name cloud_mqtt_broker --network hw03 -p 1883:1883 -ti cloud_mqtt_broker /usr/sbin/mosquitto`

3. Cloud MQTT Prcessoor

    * Build the image

        `sudo docker build -t cluod_prcessor -f dockerfile_cloud_processor .`

    * Spin up container and run `cloud_procssor.py`

        `sudo docker run --rm --name cloud_processor --network hw03 -v $PWD/:/home/ -ti cloud_processor /bin/bash /home/cloud_prcessor.sh`

4.  S3 buckets

    The S3 bucket w251-s3-bucket was created on AWS using my access_key and secret_access_key. The S3 bucket was mounted in the Ubuntu EC2 using the command below. The cloud processor dumps received images (with faces) to the folder /s3fs
    
    `sudo s3fs w251-s3-bucket /home/ubuntu/work/v3/week03/hw/s3fs/ -o passwd_file=${HOME}/.passwd-s3fs,nonempty,rw,allow_other,,mp_umask=0007,uid=1000,gid=1000`

## Submission

1. The repo for the code can be found at `https://github.com/MadhuAtBerkeley/W251/v3/tree/master/week03/hw`

2. The link to the faces can be found at https://w251-s3-bucket.s3.us-west-2.amazonaws.com/face.jpg

3. Naming of the MQTT topics: I created a two level topic for the MQTT brokers. I defined an use case for this project - detecting prisoners at a door location loc_0 and the edge device sends images with prisoner faces to MQTT forwarder/server of prison building location area_1 (geographic area). Then the forwarder send the prisoner images to the cloud.  The topic used for local broker within the prison(from door 0) is `loc_0/face_detect` and topic used for forwarding the images to the cloud (from prison location area_1) is `area_1/face_detect`

4. Choice of QoS: I picked QoS 1 for this task, which guarentees the delivery of images. The duplicates are fine in this case but need to guaranteee that every prisoner is accounted for. The QoS 1 does not have the overhead of QoS 2 and QoS 1 delivers the message much faster than QoS 2. The QoS 0 is not chosen as we might miss faces of prisoners and we would not know if they escaped or messages got lost.

