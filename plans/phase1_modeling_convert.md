- Task:
  - convert PyTorch python scripts to MindSpore python scripts
  - run validation on the generated code

- Inputs collected
  - output_folder: `mindone/transformers/models/rt_detr_v2`

- Immediate checks
  - Verify `mindone/transformers/models/rt_detr_v2` exists (create later if needed)

- Scope
  - Read all Python files under `mindone/transformers/models/rt_detr_v2/modeling_rt_detr_v2.py`
  - Treat as partially converted MindSpore scripts
  - Complete conversion with minimal modifications

- Process
  - File-by-file diffs against rules above
  - Keep variable names and structure (minimal modification)
  - Save edits in-place under `mindone/transformers/models/rt_detr_v2`
  - Record a brief change summary per file

- Human-in-the-loop
  - Surface ambiguities and request review after each non-trivial edit

- Validation checklist
  - Python AST syntax validation (parse each edited file)
  - Import resolution: ensure `import mindspore as ms`, `from mindspore import nn`, `from mindspore import mint` (as used) resolve
  - MindSpore API compatibility: scan for disallowed patterns (`.expand`, `torch.` usage, `.detach()`, `.cpu()`)
  - Cross-reference with global conversion rules
  - Quick fix of `mindone/transformers/__init__.py` for import rt_detr_v2.

- Exit criteria
  - No syntax errors; imports resolve; no prohibited APIs remain.
