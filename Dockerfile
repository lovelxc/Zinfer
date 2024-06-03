FROM --platform=linux/amd64 ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York
ENV CC=gcc
ARG CHANGE_SOURCE=true
RUN if [ ${CHANGE_SOURCE} = true ]; then \
    # sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/' /etc/apt/sources.list \
    sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list \
   # sed -i 's/security.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list \
;fi


# 系统更新以及安装编译器等工具
RUN apt-get -y update
RUN apt-get install -y build-essential cmake git wget openssh-server \
    libgflags-dev liblapack-dev libarpack2-dev libsuperlu-dev libomp-dev

# 编译安装 Eigen3
# RUN apt-get install -y libeigen3-dev
# RUN wget https://gitlab.com/libeigen/eigen/-/archive/3.4/eigen-3.4.tar.gz
# RUN tar -xzvf eigen-3.4.tar.gz
# RUN mv eigen-3.4 eigen
# RUN mkdir eigen/build
# RUN cmake eigen/ -B eigen/build/
# RUN make -C eigen/build/ -j$(nproc)
# RUN make install -C eigen/build/

# 编译安装 OpenBLAS
RUN wget https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.27/OpenBLAS-0.3.27.tar.gz
RUN tar -xzvf OpenBLAS-0.3.27.tar.gz
RUN mv OpenBLAS-0.3.27 OpenBLAS
RUN mkdir OpenBLAS/build
RUN cmake OpenBLAS/ -B OpenBLAS/build/ -DUSE_OPENMP=ON
RUN make -C OpenBLAS/build/ -j$(nproc)
RUN make install -C OpenBLAS/build/

# 编译安装 armadillo
RUN wget https://sourceforge.net/projects/arma/files/armadillo-12.8.4.tar.xz
RUN tar -vxf armadillo-12.8.4.tar.xz
RUN mkdir armadillo-12.8.4/build
# fix undefined reference to openmp related function
RUN cmake -DCMAKE_CXX_FLAGS="-fopenmp" armadillo-12.8.4/ -B armadillo-12.8.4/build/
RUN make -C armadillo-12.8.4/build -j$(nproc)
RUN make install -C armadillo-12.8.4/build

# 编译安装 googletest
RUN git clone --depth 1 https://github.com/google/googletest.git
RUN mkdir googletest/build
RUN cmake googletest/ -B googletest/build/
RUN make -C googletest/build/ -j$(nproc)
RUN make install -C googletest/build/

# 编译安装 googlebenchmark
RUN git clone --depth 1 https://github.com/google/benchmark.git
RUN mkdir benchmark/build
RUN cmake benchmark/ -B benchmark/build/ -DBENCHMARK_ENABLE_TESTING=OFF
RUN make -C benchmark/build/ -j$(nproc)
RUN make install -C benchmark/build/

# 编译安装 googlelog
RUN git clone --depth 1 https://github.com/google/glog.git
RUN mkdir glog/build
RUN cmake glog/ -B glog/build/
RUN make -C glog/build/ -j$(nproc)
RUN make install -C glog/build/

# 清理
RUN apt-get clean
RUN rm -rf OpenBLAS /googletest/ benchmark/ glog/ armadillo-12.8.4/
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /app

EXPOSE 22