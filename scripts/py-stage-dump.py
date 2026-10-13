#!/usr/bin/env python3
"""Dumps the reference text tower's per-stage tensors for the frame-70 media sequence.

The plan's section 5.4 oracle: golden stages from the validated Python implementation (mlx-vlm's
embedding_gemma2), compared against DigUp's Swift per-stage dumps. Framing: <bos> <boi> <63 rows> <eoi>
<eos> — mtmd's, verified. Writes /tmp/opencode/py-stages.json (flat lists, [t, h, d] order like Swift).
"""
import json, math, sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

MODEL = "google/embeddinggemma-2"
OUT = "/tmp/opencode/py-stages.json"

def main():
    import huggingface_hub
    from mlx_vlm.utils import load_config

    snapshot = Path(huggingface_hub.snapshot_download(
        MODEL, local_files_only=True, ignore_patterns=[".gitattributes", "README.md"]))
    config = load_config(str(snapshot))
    from mlx_vlm.models.embedding_gemma2.language import TextModel
    from mlx_vlm.models.embedding_gemma2.embedding_gemma2 import Model, ModelConfig

    model_config = ModelConfig.from_dict(config)
    text_config = model_config.text_config
    text_model = TextModel(text_config)
    weights = []
    for f in sorted(snapshot.glob("*.safetensors")):
        weights += list(mx.load(str(f)).items())
    text_weights = [(k.replace("language_model.", "", 1), v)
                    for k, v in weights if k.startswith("language_model.")]
    text_model.load_weights(text_weights, strict=True)
    mx.eval(text_model.parameters())
    cfg = text_config

    stages = {}
    # The frame-70 media sequence: bos, boi, 63 image rows, eoi, eos.
    ids = [config.get("bos_token_id", 2), config.get("boi_token_id", 255999)] \
        + [config.get("image_token_id", 258880)] * 63 \
        + [config.get("eoi_token_id", 258882), config.get("eos_token_id", 1)]
    input_ids = mx.array([ids])

    # The scaled, scattered embeddings — llama's projected rows (all 63 x 512, [t, d]).
    rows = np.fromfile("/tmp/opencode/llama-projected.bin", dtype=np.float32).reshape(63, 512)
    pad = config.get("pad_token_id", 0)
    text_ids = mx.array([[(pad if i == config.get("image_token_id", 258880) else t)
                          for i, t in enumerate(ids)]])
    emb = text_model.embed_tokens(text_ids) * mx.array(cfg.hidden_size ** 0.5)
    emb_np = np.array(emb.astype(mx.float32))[0]
    for i in range(63):
        emb_np[2 + i] = rows[i]
    inputs_embeds = mx.array(emb_np[None])

    stages["inp_scaled"] = np.array(inputs_embeds.astype(mx.float32))[0]

    # The forward, with stage capture (mirrors TextModel.__call__ exactly).
    length = inputs_embeds.shape[1]
    attention_mask = mx.ones_like(input_ids)
    full_mask = attention_mask[:, None, None, :].astype(mx.bool_)
    positions = mx.arange(length)
    position_ids = positions[None]
    sliding_mask = full_mask & (
        mx.abs(positions[:, None] - positions[None, :]) <= cfg.sliding_window
    )
    per_layer_inputs = text_model.ple(inputs_embeds)
    stages["inp_per_layer"] = np.array(per_layer_inputs.astype(mx.float32))[0]

    h = inputs_embeds
    for i, layer in enumerate(text_model.layers):
        mask = sliding_mask if cfg.layer_types[i] == "sliding_attention" else full_mask
        pli = per_layer_inputs[:, :, i]
        attn_in = text_attn_input = layer.input_layernorm(h)
        attn_out = layer.post_attention_layernorm(layer.self_attn(attn_in, mask, position_ids))
        h = attn_in * 0 + h  # keep shapes honest; the residual happens inside the layer call below
        # Recompute the layer exactly as EncoderLayer.__call__ does, capturing intermediates.
        a = layer.self_attn(layer.input_layernorm(h), mask, position_ids)
        h = h + layer.post_attention_layernorm(a)
        stages[f"attn_out-{i}"] = np.array(h.astype(mx.float32))[0]
        ffn_in = layer.pre_feedforward_layernorm(h)
        stages[f"ffn_inp_normed-{i}"] = np.array(ffn_in.astype(mx.float32))[0]
        ffn = layer.mlp(ffn_in)
        h = h + layer.post_feedforward_layernorm(ffn)
        if layer.ple_block is not None:
            gate = nn.gelu_approx(layer.ple_block.per_layer_input_gate(h)) * pli
            pl_out = h + layer.ple_block.post_per_layer_input_norm(layer.ple_block.per_layer_projection(gate))
            stages[f"per_layer_embd_out-{i}"] = np.array(
                layer.ple_block.post_per_layer_input_norm(
                    layer.ple_block.per_layer_projection(gate)).astype(mx.float32))[0]
            h = pl_out
        h = h * layer.layer_scalar
        stages[f"layer_out-{i}"] = np.array(h.astype(mx.float32))[0]

    final = text_model.embedding_projection(text_model.norm(h))
    stages["result_embd"] = np.array(final.astype(mx.float32))[0]
    pooled = np.mean(np.array(final.astype(mx.float32))[0], axis=0)
    pooled = pooled / np.linalg.norm(pooled)
    stages["pooled"] = pooled

    with open(OUT, "w") as f:
        json.dump({k: v.reshape(-1).tolist() for k, v in stages.items()}, f)

    ref = json.loads((Path(__file__).parent.parent / "Tests/LlamaRuntimeTests/Reference/vectors.json")
                     .read_text())["vectors"]["frame-70"]
    print(f"pooled vs reference frame-70: cosine {float(np.dot(pooled, np.array(ref))):.7f}", file=sys.stderr)
    print("wrote", OUT, file=sys.stderr)

def text_attn_input(x):
    return x

if __name__ == "__main__":
    main()
