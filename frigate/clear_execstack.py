# Clear the executable-stack flag (PF_X on PT_GNU_STACK) of ELF64 shared objects.
# Ubuntu 22.04's toolchain marks onnxruntime_pybind11_state.so RWE; glibc >= 2.41 (Debian 13) then refuses to dlopen it
# ("cannot enable executable stack"). Nothing in ORT needs an executable stack.
import struct, sys
for path in sys.argv[1:]:
    with open(path, "r+b") as f:
        h = f.read(64)
        assert h[:4] == b"\x7fELF" and h[4] == 2 and h[5] == 1, path
        phoff, = struct.unpack_from("<Q", h, 0x20); phentsize, phnum = struct.unpack_from("<HH", h, 0x36)
        for i in range(phnum):
            off = phoff + i * phentsize; f.seek(off); p_type, p_flags = struct.unpack("<II", f.read(8))
            if p_type == 0x6474E551 and p_flags & 1:
                f.seek(off + 4); f.write(struct.pack("<I", p_flags & ~1)); print("cleared execstack:", path)
