# DeepSeek-V3 reference model

`model.py` is DeepSeek-V3's official `inference/model.py`, copied unmodified from
[deepseek-ai/DeepSeek-V3 at commit `b15f0db`](https://github.com/deepseek-ai/DeepSeek-V3/blob/b15f0dbbbe6a4bc403306175698439ef380f5fb5/inference/model.py)
(git blob `886849975d3b146bfb84f8ea5fdff265c210fcec`). It is distributed under the MIT license in
`LICENSE-CODE`, copied from the same commit; copyright (c) 2023 DeepSeek.

`backend/deepseek_reference.py` recognises this exact file when it is imported as PyTorch source and
maps its `Transformer` onto the workbench's `LatentAttention` and `MoE` operators. `backend/test_deepseek_reference.py`:

- checks that the copy is still byte-for-byte the pinned blob (line endings aside);
- imports it through the same static path as an upload, which never executes it;
- then, as the trusted oracle, loads it in the test process with a stub `kernel` module, on the CPU in
  float32, and compares its logits with the imported graph's.

Do not edit `model.py`: the importer only recognises the pinned version.
