# Frigate (stable-tensorrt image) + an onnxruntime-gpu wheel rebuilt for Maxwell (sm_52), from ./wheel.
# Put the .whl next to this file, then:  docker build -t frigate-maxwell:stable .
ARG BASE=ghcr.io/blakeblackshear/frigate:stable-tensorrt
FROM ${BASE}
COPY onnxruntime_gpu-*.whl clear_execstack.py /tmp/ort/
RUN set -e; \
    have=$(python3 -c "import onnxruntime;print(onnxruntime.__version__)"); \
    built=$(ls /tmp/ort/*.whl | sed -E 's/.*onnxruntime_gpu-([0-9.]+)-.*/\1/'); \
    [ "$have" = "$built" ] || { echo "Frigate has onnxruntime $have, wheel is $built - rebuild the wheel with ORT_VERSION=$have"; exit 1; }; \
    PIP_BREAK_SYSTEM_PACKAGES=1 pip install --no-cache-dir --no-deps --force-reinstall /tmp/ort/*.whl; \
    D=$(python3 -c "import importlib.util as u;print(u.find_spec('onnxruntime').submodule_search_locations[0])")/capi; \
    python3 /tmp/ort/clear_execstack.py $D/*.so*; rm -rf /tmp/ort; \
    L=$D/libonnxruntime_providers_cuda.so; \
    bad=$(ldd -r $L 2>&1 | grep -E "undefined symbol|not found" | grep -v Provider_GetHost || true); \
    [ -z "$bad" ] || { echo "$bad" | head; exit 1; }; \
    python3 -c "import onnxruntime as o; print(o.__version__, o.get_available_providers())"
