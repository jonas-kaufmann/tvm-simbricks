FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04

RUN echo "deb https://repo.scala-sbt.org/scalasbt/debian all main" | tee /etc/apt/sources.list.d/sbt.list \
   && echo "deb https://repo.scala-sbt.org/scalasbt/debian /" | tee /etc/apt/sources.list.d/sbt_old.list \
   && curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | tee /etc/apt/trusted.gpg.d/sbt.asc \
   && apt-get update \
   && DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Berlin \
    apt-get install -y --no-install-recommends \
    autoconf \
    bison \
    build-essential \
    cmake \
    direnv \
    flex \
    gcc-aarch64-linux-gnu \
    help2man \
    libedit-dev \
    libfl-dev \
    libfl2 \
    libgoogle-perftools-dev \
    libtinfo-dev \
    libxml2-dev \
    llvm-dev \
    numactl \
    openjdk-11-jdk-headless \
    perl \
    perl-doc \
    python-is-python3 \
    python3 \
    python3-cloudpickle \
    python3-decorator \
    python3-dev \
    python3-numpy \
    python3-pil \
    python3-pip \
    python3-psutil \
    python3-pytest \
    python3-scipy \
    python3-setuptools \
    python3-sphinx \
    python3-sphinx-gallery \
    python3-sphinx-rtd-theme \
    python3-tornado \
    python3-typing-extensions \
    tmux \
    sbt \
    verilator \
    zlib1g \
    zlib1g-dev \
   && locale-gen en_US.UTF-8 \
   && rm -rf /var/lib/apt/lists/* \
   && echo 'eval "$(direnv hook bash)"' >> /home/vscode/.bashrc \
   && pip install --no-cache-dir \
    mxnet \
    tlcpack-sphinx-addon
RUN cd /tmp \
   && git clone --depth 1 --branch v4.228 https://github.com/verilator/verilator.git verilator \
   && cd verilator/ \
   && autoconf \
   && ./configure \
   && make -j`nproc` \
   && make install \
   && rm -rf /tmp/verilator
