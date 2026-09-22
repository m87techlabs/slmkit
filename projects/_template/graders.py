"""Domain-specific graders for this project.

A grader is a pure function (EvalPrompt, output: str) -> dict[str, float], with
known-good and known-bad fixtures in tests/. Generic graders (n-gram novelty,
log-loss, calibration, parse-rate) come from slmkit.graders instead.
"""
