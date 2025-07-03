ARG UBUNTU_RELEASE_YEAR
ARG CUDA_MAJOR
ARG CUDA_MINOR

FROM nvidia/cuda:${CUDA_MAJOR}.${CUDA_MINOR}-base-ubuntu${UBUNTU_RELEASE_YEAR}.04

ARG UBUNTU_RELEASE_YEAR
ARG CUDA_MAJOR
ARG CUDA_MINOR
ARG ZED_SDK_MAJOR
ARG ZED_SDK_MINOR

ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}compute,video,utility

RUN echo "Europe/Paris" > /etc/localtime ; echo "CUDA Version ${CUDA_MAJOR}.${CUDA_MINOR}.0" > /usr/local/cuda/version.txt

# Setup the ZED SDK
RUN apt-get update -y || true ; apt-get install --no-install-recommends lsb-release wget less zstd udev sudo python3 python3-pip libpng-dev libgomp1 -y ; \
    #python3 -m pip install --upgrade pip ; \
    python3 -m pip install numpy opencv-python ; \
    wget -q -O ZED_SDK_Linux_Ubuntu${UBUNTU_RELEASE_YEAR}.run https://download.stereolabs.com/zedsdk/${ZED_SDK_MAJOR}.${ZED_SDK_MINOR}/cu${CUDA_MAJOR}/ubuntu${UBUNTU_RELEASE_YEAR} && \
    chmod +x ZED_SDK_Linux_Ubuntu${UBUNTU_RELEASE_YEAR}.run ; ./ZED_SDK_Linux_Ubuntu${UBUNTU_RELEASE_YEAR}.run -- silent runtime_only skip_cuda && \
    rm ZED_SDK_Linux_Ubuntu${UBUNTU_RELEASE_YEAR}.run && \
    rm -rf /var/lib/apt/lists/*

# Required symlink for ZED streaming
RUN ln -sf /usr/lib/aarch64-linux-gnu/tegra/libv4l2.so.0 /usr/lib/aarch64-linux-gnu/libv4l2.so

# Install Miniconda
ENV PATH="/opt/conda/bin:$PATH"
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && rm miniconda.sh && \
    conda init && conda config --set auto_activate_base false

# Clone SONATA and install its environment
WORKDIR /root
RUN git clone --recurse-submodules https://github.com/facebookresearch/sonata.git

WORKDIR /root/sonata

# Create SONATA Conda environment
RUN conda env create -f environment.yml && \
    echo "conda activate sonata" >> ~/.bashrc

# Default working directory
WORKDIR /usr/local/zed

# Final entrypoint
CMD ["/bin/bash"]
