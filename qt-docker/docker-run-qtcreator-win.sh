#! /bin/sh
#

# Fix for running GUI programs from within the docker container:
#
# Make xorg disable access control, i.e. let any x client connect to our
# server.
#xhost +

# In order to let the container connect to the pulseaudio server over TCP,
# the module-native-protocol-tcp module must be enabled:
#
# load-module module-native-protocol-tcp

# 172.17.0.1 is the default host ip address of the docker network interface
# (docker0). In case this is changed (e.g. docker container not using NAT
# address) the below env var must of course be updated.
set-variable -name DISPLAY -value 172.24.240.1:0.0
set-variable -name PULSE_SERVER_TCP_ENV -value "-e PULSE_SERVER=tcp:172.17.0.1:4713"


# Create the container with all options needed to let a GUI program
# like qtcreator connect to the host xorg server.
# The current user home directory is also mounted directly into the
# container.
docker run -it --privileged -v //c/Users/brai/:/home/watad -e DISPLAY=$DISPLAY $PULSE_SERVER_TCP_ENV --name qtcreator --rm -d  --network host wt-vision/qt qtcreator
