# Smoke test: runs a tiny Conv model on CPU and on CUDA and compares the results.
# get_available_providers() lists CUDA even when its kernels cannot run on your GPU - this does not lie.
# Inside the container:  python3 gputest.py   (needs numpy + onnx; both are in the Immich and Frigate images)
import numpy as np, onnx, onnxruntime as ort
from onnx import helper, TensorProto
X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 64, 64])
Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)
W = helper.make_tensor("w", TensorProto.FLOAT, [8, 3, 3, 3], np.random.rand(8*27).astype(np.float32))
g = helper.make_graph([helper.make_node("Conv", ["x", "w"], ["c"]), helper.make_node("Relu", ["c"], ["r"]),
                       helper.make_node("ReduceMean", ["r"], ["y"], keepdims=0)], "t", [X], [Y], [W])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]); m.ir_version = 9
x = np.random.rand(1, 3, 64, 64).astype(np.float32)
cpu = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
s = ort.InferenceSession(m.SerializeToString(), providers=["CUDAExecutionProvider"])
print("providers:", s.get_providers())
gpu = s.run(None, {"x": x})[0]
print("cpu", cpu, "gpu", gpu, "match", np.allclose(cpu, gpu, rtol=1e-3))
