# onnxruntime-gpu WHEEL for Frigate (stable-tensorrt image), rebuilt for Maxwell (GTX 970, sm_52).
# Produces only the .whl; ../Dockerfile installs it into the Frigate image.
# Build:  docker build --build-arg JOBS=3 --target export -o out .   (then copy out/*.whl to ../)
# Rebuild when Frigate bumps onnxruntime: pass --build-arg ORT_VERSION=x.y.z (../Dockerfile checks it matches).
ARG ORT_VERSION=1.24.4
ARG PY=3.11

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS build
ARG ORT_VERSION
ARG PY
ARG JOBS=3
ARG EXTRA_BUILD_ARGS=
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates build-essential && rm -rf /var/lib/apt/lists/*
RUN uv venv -p ${PY} /opt/buildenv && VIRTUAL_ENV=/opt/buildenv uv pip install "numpy<2" packaging setuptools wheel "cmake<4.1" ninja
ENV PATH=/opt/buildenv/bin:$PATH
RUN git clone --depth 1 --branch v${ORT_VERSION} --recursive --shallow-submodules https://github.com/microsoft/onnxruntime /src
WORKDIR /src
# Known bugs in newer onnxruntime (1.27.x) when building for old GPUs only - applied only if the code has them.
RUN f=onnxruntime/core/providers/cuda/llm/attention.cc; \
    if grep -q '::onnxruntime::contrib::cuda::kCutlassSafeMaskFilterValue' $f 2>/dev/null; then \
      sed -i 's/::onnxruntime::contrib::cuda::kCutlassSafeMaskFilterValue/-1.0e+30f/' $f && echo "patched attention.cc"; fi; \
    c=cmake/onnxruntime_providers_cuda.cmake; \
    if grep -q 'onnxruntime_filter_cuda_archs(_ort_llm_cuda_architectures MIN_SM 75' $c; then \
      sed -i '/onnxruntime_filter_cuda_archs(_ort_llm_cuda_architectures MIN_SM 75/a\        if(NOT _ort_llm_cuda_architectures)\n          set(_ort_llm_cuda_architectures "75-virtual")\n        endif()' $c && echo "patched llm archs"; fi; \
    true
RUN ./build.sh --config Release --build_wheel --skip_tests --allow_running_as_root \
      --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr \
      --cmake_generator Ninja --parallel ${JOBS} --nvcc_threads 1 --compile_no_warning_as_error \
      ${EXTRA_BUILD_ARGS} \
      --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=52 \
        onnxruntime_BUILD_UNIT_TESTS=OFF \
        onnxruntime_USE_FLASH_ATTENTION=OFF \
        onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION=OFF \
        onnxruntime_USE_LEAN_ATTENTION=OFF \
 && mkdir /out && cp build/Linux/Release/dist/*.whl /out/
# Self-check: the CUDA provider library must have no missing symbols (otherwise the CUDA EP silently fails to load).
RUN bad=$(ldd -r build/Linux/Release/libonnxruntime_providers_cuda.so 2>&1 | grep "undefined symbol" | grep -v Provider_GetHost || true); \
    [ -z "$bad" ] || { echo "CUDA provider library will not load:"; echo "$bad" | head -20; exit 1; }; echo "symbol check OK"

FROM scratch AS export
COPY --from=build /out/ /
