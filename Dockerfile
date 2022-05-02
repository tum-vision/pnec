FROM ubuntu:20.04

# install libraries
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Lnndon
RUN apt-get update && apt-get install -y sudo git cmake build-essential
# suitesparse, boost, eigen 3.4!!!, ceres dependencies, opencv
RUN apt-get update && \
    apt-get install -y \
    libgoogle-glog-dev libgflags-dev libatlas-base-dev \
    libsuitesparse-dev \
    libboost-all-dev \
    libeigen3-dev \
    libopencv-dev python3-opencv
# ceres
WORKDIR /home/
RUN git clone -b '1.13.0' https://ceres-solver.googlesource.com/ceres-solver && cd ./ceres-solver/ && cmake . -DEXPORT_BUILD_DIR=ON && make -j8 install

# basalt (opengv is included in basalt) from gitlab
WORKDIR /app/pnec/third_party
RUN git clone --recursive https://gitlab.com/VladyslavUsenko/basalt.git && \
# # get the latest version of magic enum, to avoid compiling errors
    cd basalt/thirdparty/ && \
    rm -r magic_enum && \
    git clone https://github.com/Neargye/magic_enum.git && \
    cd .. && \
    ./scripts/install_deps.sh

COPY . /app/pnec/
WORKDIR /app/pnec
RUN mkdir build && cd build && cmake .. && make -j8
COPY kitti_docker.sh /app/kitti_docker.sh