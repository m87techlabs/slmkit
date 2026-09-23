"""Template project. Copy this directory to start a new LLM:

    cp -r projects/_template projects/<name>

Then implement the abstract methods below and delete this docstring.
See docs/DESIGN.md 6.4 and the checklist in CONTRIBUTING.md.
"""

from __future__ import annotations

# from pydantic import BaseModel, ConfigDict
# from slmkit.project_api import Doc, EvalPrompt, Grader, Project, TokenizerSpec
# from slmkit.registry import register_project


# class TemplateArgs(BaseModel):
#     model_config = ConfigDict(extra="forbid")   # typos in the YAML fail loudly
#     ...                                          # everything under `project.args:`
#
# @register_project("<name>")
# class TemplateProject(Project):
#     Args = TemplateArgs
#     data_version = 1                        # bump when documents()/augment() output changes
#
#     def ingest(self, raw_dir): ...          # idempotent download
#     def documents(self, raw_dir): ...       # yield Doc(group=<leakage-free key>)
#     def tokenizer_spec(self): ...           # char | bpe(vocab) | fixed(list)
#     def eval_prompts(self, split): ...
#     def graders(self): ...                  # domain graders only; generic ones come
#                                             # from slmkit.graders
#
#     Optional: augment (train split only), split_exclusions, sft_examples, logits_processor.
#     See projects/shakespeare_char/project.py for a complete, small example.
