FROM ubuntu:20.04

# install libraries
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Lnndon
RUN apt-get update && apt-get install -y sudo git cmake build-essential
# suitesparse, boost, eigen 3.4!!!, ceres dependencies, opencv
# RUN git clone https://gitlab.com/libeigen/eigen.git && \
#     cd eigen/ && \
#     mkdir build && \
#     cd build && \
#     cmake .. && \
#     make install && \
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
    # mkdir build && \
    # cd build/ && \
#     cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_FLAGS="-Wno-error" && \
#     make -j8

# # WORKDIR /app/pnec/thrid_party/basalt/thirdparty
# # RUN 
# # RUN cd ./basalt/ && ./scripts/install_deps.sh  && mkdir build && cd build/ && cmake -E env CXXFLAGS="-Wno-error" && cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo && make -j8
# WORKDIR /app/pnec/thrid_party
# RUN 

COPY . /app/pnec/
