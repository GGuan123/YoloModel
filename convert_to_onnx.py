"""Convert Darknet YOLOv3 (cfg+weights) to ONNX. Output: yolov3.onnx"""
import os, sys, struct
import numpy as np

# Stub broken onnx.defs (Windows Store Python path issue)
import types as _t
_d = _t.ModuleType("onnx.defs")
_d.OpSchema = type("O", (), {})
_d.SchemaError = type("E", (Exception,), {})
_d.get_all_schemas_with_history = lambda: []
_d.get_schema = lambda *a, **k: None
_d.has = lambda *a, **k: False
_d.onnx_opset_version = lambda: 21
sys.modules["onnx.defs"] = _d
sys.modules["onnx.checker"] = _t.ModuleType("onnx.checker")

import onnx
from onnx import helper, TensorProto

DIR = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(DIR, "yolov3.cfg")
WTS = os.path.join(DIR, "yolov3.weights")
OUT = os.path.join(DIR, "yolov3.onnx")
IH, IW, OPSET = 608, 608, 11

def parse_cfg(p):
    blocks = []
    cur = None
    for ln in open(p).readlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        if ln.startswith("["):
            if cur:
                blocks.append(cur)
            cur = {"type": ln[1:-1]}
        else:
            k, v = ln.split("=", 1)
            cur[k.strip()] = v.strip()
    if cur:
        blocks.append(cur)
    return blocks

def load_wts(p):
    with open(p, "rb") as f:
        f.read(20)  # header
        return np.frombuffer(f.read(), dtype=np.float32)

def assign_wts(blocks, data):
    ptr = 0
    och = {-1: 3}
    for idx, b in enumerate(blocks):
        if b["type"] != "convolutional":
            if b["type"] == "route":
                ls = [int(x.strip()) for x in b["layers"].split(",")]
                s = sum(och.get(l if l >= 0 else idx + l, 3) for l in ls)
                och[idx] = s
            elif b["type"] in ("shortcut", "maxpool", "avgpool", "upsample", "softmax", "yolo"):
                och[idx] = och.get(idx - 1, 3)
            continue
        f = int(b["filters"]); sz = int(b["size"]); bn = b.get("batch_normalize", "0") == "1"
        ic = och.get(idx - 1, 3)
        b["bias"] = data[ptr: ptr + f].copy(); ptr += f
        if bn:
            b["bn_s"] = data[ptr: ptr + f].copy(); ptr += f
            b["bn_m"] = data[ptr: ptr + f].copy(); ptr += f
            b["bn_v"] = data[ptr: ptr + f].copy(); ptr += f
        wc = f * ic * sz * sz
        b["w"] = data[ptr: ptr + wc].copy(); ptr += wc
        b["ic"] = ic; b["f"] = f; b["sz"] = sz
        och[idx] = f
    return blocks

def build(blocks):
    ns, init = [], []
    ins = [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, IH, IW])]
    vi, lo = {"input": [1, 3, IH, IW]}, {-1: "input"}
    c = [0]
    def nn(p):
        c[0] += 1; return f"{p}_{c[0]}"
    def ai(nm, arr):
        if arr.dtype == np.int64:
            init.append(helper.make_tensor(nm, TensorProto.INT64, arr.shape, arr.flatten().tolist()))
        else:
            init.append(helper.make_tensor(nm, TensorProto.FLOAT, arr.shape, arr.astype(np.float32).flatten().tolist()))

    for idx, b in enumerate(blocks):
        bt = b["type"]; inp = lo.get(idx - 1, "input")

        if bt == "convolutional":
            f, sz = b["f"], b["sz"]; stride = int(b.get("stride", "1"))
            pad = int(b.get("pad", "0")); bn = "bn_s" in b; act = b.get("activation", "linear")
            w = b["w"].reshape(f, b["ic"], sz, sz)
            if bn:
                eps = 1e-6; sc, mn, vr, bi = b["bn_s"], b["bn_m"], b["bn_v"], b["bias"]
                std = np.sqrt(vr + eps)
                fw = w * (sc / std).reshape(-1, 1, 1, 1); fb = bi - (sc * mn) / std
            else:
                fw, fb = w, b["bias"]

            if pad:
                p = sz // 2
                pn = nn("padval"); ai(pn, np.array([0, 0, p, p, 0, 0, p, p], dtype=np.int64))
                pn2 = nn("pad"); ns.append(helper.make_node("Pad", [inp, pn], [pn2], mode="constant"))
                inp = pn2
            wn, bn2 = nn("W"), nn("B")
            ai(wn, fw.astype(np.float32)); ai(bn2, fb.astype(np.float32))
            cv = nn("conv")
            ns.append(helper.make_node("Conv", [inp, wn, bn2], [cv], kernel_shape=[sz, sz], strides=[stride, stride]))
            ch, hh, ww = vi.get(inp, [1, 3, IH, IW])[1], vi.get(inp, [1, 3, IH, IW])[2] // stride, vi.get(inp, [1, 3, IH, IW])[3] // stride
            vi[cv] = [1, f, hh, ww]; out = cv
            if act == "leaky":
                an = nn("leaky"); ns.append(helper.make_node("LeakyRelu", [out], [an], alpha=0.1))
                out = an; vi[out] = vi[cv]
            elif act != "linear":
                an = nn("relu"); ns.append(helper.make_node("Relu", [out], [an]))
                out = an; vi[out] = vi[cv]
            lo[idx] = out

        elif bt == "maxpool":
            s = int(b.get("size", "2")); st = int(b.get("stride", str(s)))
            mp = nn("mp"); ns.append(helper.make_node("MaxPool", [inp], [mp], kernel_shape=[s, s], strides=[st, st]))
            vi[mp] = [1, vi[inp][1], vi[inp][2] // st, vi[inp][3] // st]
            lo[idx] = mp

        elif bt == "shortcut":
            fr = int(b["from"]); si = idx + fr if fr < 0 else fr; sr = lo.get(si, inp)
            ad = nn("add"); ns.append(helper.make_node("Add", [inp, sr], [ad])); vi[ad] = vi[inp]; out = ad
            if b.get("activation", "linear") == "leaky":
                an = nn("leaky"); ns.append(helper.make_node("LeakyRelu", [out], [an], alpha=0.1)); out = an
            lo[idx] = out

        elif bt == "route":
            ls = [int(x.strip()) for x in b["layers"].split(",")]
            srcs = [lo.get(l if l >= 0 else idx + l, inp) for l in ls]
            if len(srcs) == 1:
                lo[idx] = srcs[0]
            else:
                cc = nn("cat"); ns.append(helper.make_node("Concat", srcs, [cc], axis=1))
                vi[cc] = [1, sum(vi[s][1] for s in srcs), vi[srcs[0]][2], vi[srcs[0]][3]]
                lo[idx] = cc

        elif bt == "upsample":
            st = int(b.get("stride", "2")); up = nn("up"); scn = nn("scales")
            ai(scn, np.array([1.0, 1.0, float(st), float(st)], dtype=np.float32))
            ns.append(helper.make_node("Resize", [inp, "", scn], [up], mode="nearest"))
            vi[up] = [1, vi[inp][1], vi[inp][2] * st, vi[inp][3] * st]
            lo[idx] = up

        elif bt in ("yolo", "avgpool", "softmax"):
            if bt == "avgpool":
                ap = nn("ap"); ns.append(helper.make_node("GlobalAveragePool", [inp], [ap])); vi[ap] = [1, vi[inp][1], 1, 1]; lo[idx] = ap
            elif bt == "softmax":
                sm = nn("sm"); ns.append(helper.make_node("Softmax", [inp], [sm], axis=1)); vi[sm] = vi[inp]; lo[idx] = sm
            else:
                lo[idx] = inp

    outs = []
    for idx, b in enumerate(blocks):
        if b["type"] == "yolo":
            on = lo.get(idx - 1, "input")
            outs.append(helper.make_tensor_value_info(on, TensorProto.FLOAT, vi[on]))
    if not outs:
        on = list(lo.values())[-1]
        outs = [helper.make_tensor_value_info(on, TensorProto.FLOAT, vi.get(on, [1, 3, 1, 1]))]

    g = helper.make_graph(ns, "yolov3", ins, outs, initializer=init)
    m = helper.make_model(g, producer_name="darknet2onnx", opset_imports=[helper.make_opsetid("", OPSET)])
    m.ir_version = 7
    return m

if __name__ == "__main__":
    print("Parsing cfg..."); bl = parse_cfg(CFG); print(f"  {len(bl)} layers")
    print("Loading weights..."); dt = load_wts(WTS); print(f"  {len(dt)} values")
    print("Assigning..."); bl = assign_wts(bl, dt)
    print("Building ONNX..."); m = build(bl)
    print(f"Saving {OUT}..."); onnx.save(m, OUT)
    print(f"Done: {os.path.getsize(OUT)/1024/1024:.1f} MB")
