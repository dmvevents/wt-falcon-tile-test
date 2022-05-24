FROM gcr.io/dulcet-nucleus-331721/x86/tf-base:latest


# copy pro file for qmake
ADD . /usr/src/server
COPY wt-sadata-vision.pro /usr/src/server/wt-sadata-vision.pro
COPY detect.mp4 /usr/src/server/detect.mp4



WORKDIR /usr/src/server/build
RUN  qmake ../wt-sadata-vision.pro
RUN make 

VOLUME ["/app-data"]

EXPOSE 502 8080 8000 3000

#CMD ["./ids-peak-sim"]
CMD ["/bin/bash"]
