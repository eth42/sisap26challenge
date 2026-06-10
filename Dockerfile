FROM python:3.13.5-slim

WORKDIR /app

### System setup ###
RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential
RUN apt-get install -y --no-install-recommends curl
RUN apt-get install -y --no-install-recommends git-all
RUN apt-get install -y --no-install-recommends libopenblas-dev
RUN rm -rf /var/lib/apt/lists/*

### Rust setup ###
# Set environment variable for Rust version
ENV RUST_VERSION=1.85.0
# Install rustup with the specific Rust version
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain $RUST_VERSION
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustc --version

### Python requirements ###
# Copy requirements
COPY requirements.txt .
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

### Required repositories ###
# HIOB
RUN git clone --branch affine-hiob2 https://github.com/eth42/hiob.git
RUN cd hiob && maturin build -r
RUN pip install hiob/target/wheels/*.whl
# Graph Index Libraries
ENV GIA_HASH=498eb45f80ee06da6fb82581588c7ae1e1daa5ee
ENV GIB_HASH=88a7c85eda27ae81637d413a2486a18e5c091473
RUN git clone https://github.com/eth42/GraphIndexAPI.git
RUN cd GraphIndexAPI && git checkout $GIA_HASH
RUN git clone https://github.com/eth42/GraphIndexBaselines.git
RUN cd GraphIndexBaselines && git checkout $GIB_HASH
RUN cd GraphIndexBaselines && maturin build -r --features=pyprec16,pyref32
RUN pip install GraphIndexBaselines/target/wheels/*.whl

# Force rebuild of all checkpoints from this point on upon change
ENV _DIRT=1

# Copy source code
COPY . .
RUN chmod +x *.py
