"""E3: learned null conditioning for Stable Audio Open (PROJECT_DESIGN.md §1.2, §5.3).

Only the representation of "no condition" is learned; the prompt text is never touched and
this is not a negative prompt. Heavy dependencies (torch, stable-audio-tools) are imported lazily.
"""
