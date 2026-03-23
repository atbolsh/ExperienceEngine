---
description: Output and debugging conventions for this repo
globs: "**/*.py"
---

This repo is a developer tool, not an end-user product. All output should
bias for **clarity over prettiness**:

- Print full model output including all special tokens (<think>, <tool_call>,
  etc.) — never truncate or abbreviate debug output.
- Log every tool call invocation and its result in full.
- When in doubt, show more information, not less.
