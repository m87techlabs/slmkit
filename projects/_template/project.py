"""Template project. Copy this directory to start a new LLM:

    cp -r projects/_template projects/<name>

Then implement the abstract methods below and delete this docstring.
See docs/DESIGN.md 6.4 and the checklist in CONTRIBUTING.md.
"""

from __future__ import annotations

# from slmkit.project_api import Doc, EvalPrompt, Grader, Project, TokenizerSpec
# from slmkit.registry import register_project


# @register_project("<name>")
# class TemplateProject(Project):
#     name = "<name>"
#
#     class Args(BaseModel):
#         ...
#
#     def ingest(self, raw_dir): ...          # idempotent download
#     def documents(self, raw_dir): ...       # yield Doc(group=<leakage-free key>)
#     def tokenizer_spec(self): ...           # char | bpe(vocab) | fixed(list)
#     def eval_prompts(self, split): ...
#     def graders(self): ...                  # domain graders only; generic ones come
#                                             # from slmkit.graders
