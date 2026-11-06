# onnxruntime-gpu for Maxwell GPUs (GTX 9xx, sm_52) in Immich and Frigate

Recent `onnxruntime-gpu` wheels no longer include kernels for **Maxwell** (compute capability
5.2: GTX 950/960/970/980/Titan X, some Quadro M cards). The CUDA execution provider still
*loads*, and `get_available_providers()` still lists it, but the first inference fails with

```
cudaErrorNoKernelImageForDevice: no kernel image is available for execution on the device
```

and Immich/Frigate fall back to the CPU, or fail outright.

The CUDA, cuDNN 9 and cuBLAS 12 libraries in the official images still support Maxwell.
**Only onnxruntime is missing the sm_52 code.** This repo rebuilds onnxruntime with
`CMAKE_CUDA_ARCHITECTURES=52` and drops it into the official images. Nothing else changes.

These are recipes (Dockerfiles), not prebuilt images: you build them yourself and rebuild when
upstream moves to a new onnxruntime.

## Verified setup

| | |
|---|---|
| GPU | GeForce GTX 970 (4 GB), NVIDIA driver 580 |
| Host | Ubuntu, Docker + NVIDIA Container Toolkit |
| Immich | `immich-machine-learning:v3-cuda` (server 3.3.x) + onnxruntime **1.27.1** built for sm_52 |
| Frigate | `stable-tensorrt` (0.18.x) + onnxruntime **1.24.4** built for sm_52, ONNX detector, YOLOv9-s 320 |

Results on that card:

- **Immich**: smart search (CLIP) and face recognition run on the CUDA EP. About 780 MiB VRAM while models are loaded.
- **Frigate**: YOLOv9-s-320 at about **10–14 ms** per inference on the CUDA EP (vs about 32 ms on the CPU), about 210 MiB VRAM. NVDEC decoding (`preset-nvidia`) works too.

Other Maxwell cards should behave the same. Older or newer architectures probably need a
different `CMAKE_CUDA_ARCHITECTURES` value and are untested here.

## Immich

```sh
docker pull ghcr.io/immich-app/immich-machine-learning:v3-cuda
cd immich
docker build --build-arg JOBS=3 -t immich-ml-maxwell:v3 .
```

Then point the `immich-machine-learning` service at `immich-ml-maxwell:v3` and give it the GPU.
See [immich/compose.example.yaml](immich/compose.example.yaml).

The build compiles onnxruntime inside `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`, then swaps
the `onnxruntime` package inside the official image. The final stage **fails on purpose** when:

- the base image's onnxruntime version differs from `ORT_VERSION` (then rebuild with
  `--build-arg ORT_VERSION=<the version it reports>`), or
- `libonnxruntime_providers_cuda.so` has unresolved symbols, which would make the CUDA EP
  fail to load at runtime.

The image's TensorRT-RTX execution provider package is removed. It needs a much newer GPU anyway.

## Frigate

Frigate needs two steps, because its image uses a different Python and base OS than the
build container:

```sh
# 1. Build the wheel. PY must match Frigate's Python (3.11 for 0.18) and ORT_VERSION its onnxruntime.
cd frigate/wheel
docker build --build-arg JOBS=3 --build-arg ORT_VERSION=1.24.4 --target export -o out .
cp out/onnxruntime_gpu-*.whl ..      # keep only one wheel next to ../Dockerfile

# 2. Install it into the official image.
docker pull ghcr.io/blakeblackshear/frigate:stable-tensorrt
cd ..
docker build -t frigate-maxwell:stable .
```

Use `frigate-maxwell:stable` as the image, give it the GPU, and use the ONNX detector.
See [frigate/compose.example.yaml](frigate/compose.example.yaml) and
[frigate/config.example.yml](frigate/config.example.yml).
[frigate/model](frigate/model) builds the YOLOv9 ONNX model following Frigate's documented
export, with CPU-only PyTorch so the build is smaller.

To find the versions to build for:

```sh
docker run --rm --entrypoint python3 ghcr.io/blakeblackshear/frigate:stable-tensorrt \
  -c "import sys, onnxruntime; print(sys.version.split()[0], onnxruntime.__version__)"
```

## Gotchas this repo already handles

1. **onnxruntime ≥ 1.27: build errors when targeting only pre-Turing GPUs.** Two upstream bugs:
   - `llm/attention.cc` uses `kCutlassSafeMaskFilterValue`, which only exists when
     memory-efficient attention is enabled. It's patched to the literal value (`-1.0e+30f`).
   - With only archs below sm_75, CMake skips the LLM/MoE CUTLASS object library, but the MoE
     ops still reference it. That gives `undefined symbol: ...MoeGemmRunner...getConfigs`, and the
     CUDA EP **silently fails to load**. The library is built as `75-virtual` (PTX only) instead.
     The symbols resolve, and those kernels never run on Maxwell. Nothing in Immich or Frigate uses MoE ops.

   Both patches are applied only if the source still has the bug. The `grep -q` checks in the
   Immich Dockerfile fail the build if upstream changes the code, so you'll notice.
2. **Executable-stack flag vs. glibc 2.41 (Debian 13, which Frigate uses).** A wheel built on
   Ubuntu 22.04 marks `onnxruntime_pybind11_state.so` as needing an executable stack, and newer
   glibc refuses to load it (`cannot enable executable stack`).
   [frigate/clear_execstack.py](frigate/clear_execstack.py) clears the `PF_X` bit on
   `PT_GNU_STACK`. onnxruntime doesn't need an executable stack.
3. **Flash, memory-efficient and lean attention are disabled.** They need newer GPUs anyway.
4. **`get_available_providers()` lies.** It only says the provider library exists, not that
   its kernels can run on your GPU. Check for real with:
   ```sh
   docker exec -it <container> python3 - < tools/gputest.py
   ```
   It should print `providers: ['CUDAExecutionProvider', ...]` and `match True`.
   For Frigate, the detector inference speed in *System metrics* is the easiest check.

## When to rebuild

Whenever Immich or Frigate ship a new onnxruntime version. The builds fail loudly on a version
mismatch, so a plain `docker compose pull` won't silently put you back on the CPU. Pull the
new base image, rebuild with the new `ORT_VERSION`, and recreate the container.

Compiling onnxruntime takes roughly 45–60 minutes with `JOBS=3` on a 4-core machine, and the
Docker build cache grows by about 25 GB. Run `docker builder prune` afterwards.

## Rolling back

Point the service back at the official image (`...machine-learning:v3-cuda` or
`...frigate:stable-tensorrt`, or the CPU image) and recreate the container. Nothing on disk
depends on the custom build.

## License

MIT for the files in this repo. onnxruntime is MIT-licensed by Microsoft. Immich, Frigate and the
NVIDIA CUDA images keep their own licenses; this repo doesn't redistribute any of them.
Not affiliated with the Immich, Frigate or onnxruntime projects.
