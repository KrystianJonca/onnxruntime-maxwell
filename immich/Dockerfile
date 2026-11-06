# Immich ML (CUDA) with onnxruntime-gpu rebuilt for Maxwell (GTX 970, sm_52).
# Stock wheels drop sm_52 -> cudaErrorNoKernelImageForDevice. cuDNN 9.10 / cuBLAS 12 in the base still support Maxwell.
# Rebuild whenever Immich bumps onnxruntime: set ORT_VERSION to match (the final stage checks it).
# Build:  docker build --build-arg JOBS=3 -t immich-ml-maxwell:v3 .   (JOBS = parallel compile jobs; lower it if the build runs out of RAM)
ARG ORT_VERSION=1.27.1
ARG BASE=ghcr.io/immich-app/immich-machine-learning:v3-cuda

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS build
ARG ORT_VERSION
ARG JOBS=3
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates build-essential && rm -rf /var/lib/apt/lists/*
RUN uv venv -p 3.13 /opt/buildenv && VIRTUAL_ENV=/opt/buildenv uv pip install numpy packaging setuptools wheel "cmake<4.1" ninja
ENV PATH=/opt/buildenv/bin:$PATH
RUN git clone --depth 1 --branch v${ORT_VERSION} --recursive --shallow-submodules https://github.com/microsoft/onnxruntime /src
WORKDIR /src
# 1.27.1 bug: kCutlassSafeMaskFilterValue is only defined with memory-efficient attention ON, but llm/attention.cc uses it unconditionally.
RUN sed -i 's/::onnxruntime::contrib::cuda::kCutlassSafeMaskFilterValue/-1.0e+30f/' onnxruntime/core/providers/cuda/llm/attention.cc \
 && grep -q -- '-1.0e+30f);' onnxruntime/core/providers/cuda/llm/attention.cc
# 1.27.1 bug: with only pre-Turing archs, the LLM (MoE/cutlass) object library is skipped entirely, but the MoE ops still
# reference it -> "undefined symbol: ...MoeGemmRunner...getConfigs" and the CUDA EP fails to load. Build it as compute_75
# PTX only: symbols resolve, kernels can never run on sm_52 (nothing in Immich uses MoE ops).
RUN sed -i '/onnxruntime_filter_cuda_archs(_ort_llm_cuda_architectures MIN_SM 75/a\        if(NOT _ort_llm_cuda_architectures)\n          set(_ort_llm_cuda_architectures "75-virtual")\n        endif()' cmake/onnxruntime_providers_cuda.cmake \
 && grep -q 'set(_ort_llm_cuda_architectures "75-virtual")' cmake/onnxruntime_providers_cuda.cmake
RUN ./build.sh --config Release --build_wheel --skip_tests --allow_running_as_root \
      --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr \
      --cmake_generator Ninja --parallel ${JOBS} --nvcc_threads 1 --compile_no_warning_as_error \
      --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=52 \
        onnxruntime_BUILD_UNIT_TESTS=OFF \
        onnxruntime_USE_FLASH_ATTENTION=OFF \
        onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION=OFF \
        onnxruntime_USE_LEAN_ATTENTION=OFF \
 && mkdir /out && cp build/Linux/Release/dist/*.whl /out/

FROM ${BASE}
ARG ORT_VERSION
COPY --from=build /out/ /tmp/ort/
RUN set -e; SP=/opt/venv/lib/python3.13/site-packages; \
    have=$(python3 -c "import onnxruntime;print(onnxruntime.__version__)"); \
    [ "$have" = "$ORT_VERSION" ] || { echo "base image has onnxruntime $have, built $ORT_VERSION - bump ORT_VERSION"; exit 1; }; \
    rm -rf $SP/onnxruntime $SP/onnxruntime_ep_nv_tensorrt_rtx $SP/onnxruntime_ep_nv_tensorrt_rtx*.dist-info; \
    python3 -c "import zipfile,glob,sys; z=zipfile.ZipFile(glob.glob('/tmp/ort/*.whl')[0]); [z.extract(n, sys.argv[1]) for n in z.namelist() if n.startswith('onnxruntime/')]" $SP; \
    rm -rf /tmp/ort; \
    bad=$(ldd -r $SP/onnxruntime/capi/libonnxruntime_providers_cuda.so 2>&1 | grep -E "undefined symbol|not found" | grep -v Provider_GetHost || true); \
    [ -z "$bad" ] || { echo "CUDA provider library will not load:"; echo "$bad" | head -20; exit 1; }; \
    python3 -c "import onnxruntime as o; print(o.__version__, o.get_available_providers())"
