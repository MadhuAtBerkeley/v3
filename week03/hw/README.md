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

    This spins up the forwarder that connects with both broker, so make sure the broker is up on VSI as well (see below).

    * Build the image

        `sudo docker build -t nx_mqtt_forwarder -f dockerfile_mqtt_forwarder .`

    * Spin up container and run `nx_forwarder.py`

        `sudo docker run --rm --name nx_mqtt_forwarder --network hw03 -v ~/work/W251/v3/week03/hw/:/home/ -ti nx_mqtt_forwarder /bin/sh /home/nx_forwarder.sh`

4. OpenCV face detector

    * Build the image

        `sudo docker build -t nx_face_detector -f dockerfile_face_detector .`

    * Spin up the container and run `nx_face_detector.py`

        `sudo docker run --rm --privileged -e DISPLAY --name nx_face_detector --network hw03 -v ~/work/W251/v3/week03/hw/:/home/ -ti nx_face_detector /bin/bash /home/nx_face_detector.sh`

    * Note that in `nx_face_detector.py` I added in a timeout of 5 seconds as well as a single digit counter to keep track of faces/pictures. These were used just so that I don't overwhelm the output with tons of images. Tests have been done with these two restrictions removed and it still works nicely. 

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

        Unfortunately, I didn't find a way to decode the bytes message without openCV, so this receiver image is the same with the image for face detector, except that I also needed to add `s3cmd`. It is bigger than I idealy wanted, but it works for now.

        `sudo docker build -t cluod_prcessor -f dockerfile_cloud_processor .`

    * Spin up container and run `cloud_procssor.py`

        `sudo docker run --rm --name cloud_processor --network hw03 -v $PWD/:/home/ -ti cloud_processor /bin/bash /home/cloud_prcessor.sh`

4.  S3 buckets

    The S3 bucket w251-s3-bucket was created on AWS using my access_key and secret_access_key. The S3 bucket was mounted in the Ubuntu EC2 using the command below.
    
    `sudo s3fs w251-s3-bucket /home/ubuntu/work/v3/week03/hw/s3fs/hw3_data -o passwd_file=${HOME}/.passwd-s3fs,nonempty,rw,allow_other,,mp_umask=0007,uid=1000,gid=1000`

## Submission

1. The repo for the code can be found at `https://github.com/MadhuAtBerkeley/W251/v3/tree/master/week03/hw`, please let me know if there is any trouble accessing it.

2. The link to the faces can be found at https://w251-s3-bucket.s3.us-west-2.amazonaws.com/face.jpg

3. Naming of the MQTT topics: I created a simple single-level topic for the MQTT topic .

4. Choice of QoS: I picked QoS 0 for this task, which is also commonly known as "fire and forgot". 
