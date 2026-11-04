"""Pairwise judges for E4 (PROJECT_DESIGN.md §5.4).

Every judge implements ``compare(path_A, path_B, text, question) -> "A" | "B" | "tie"``, i.e. a
*positional* answer for one presentation; ``run.py`` presents every pair in both orders and
``pairs.to_relative`` maps answers back to (w_a, w_b). The Qwen judge gets the same three
questions as listeners (Polish + English, ``templates/pairwise.txt``) and hears clips normalised
to −23 LUFS like the listeners. Heavy imports (torch, transformers) are lazy.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from audiocfg.schema import QUESTIONS

TEMPLATE_PATH = Path(__file__).with_name("templates") / "pairwise.txt"


# ----------------------------------------------------------------------------- template
def load_template(path: str | Path = TEMPLATE_PATH) -> dict[str, str]:
    """Parse ``[section]`` blocks of the template file into ``{section: text}``."""
    sections: dict[str, list[str]] = {}
    current = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            continue
        m = re.fullmatch(r"\[([\w.]+)\]\s*", line)
        if m:
            current = m.group(1)
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def render_prompt(text: str, question: str, template: dict[str, str] | None = None) -> str:
    """User message for one presentation: description + question in English and Polish."""
    t = template or load_template()
    if question not in QUESTIONS:
        raise ValueError(f"unknown question {question!r}")
    return t["user"].format(text=text, question_en=t[f"question.{question}.en"],
                            question_pl=t[f"question.{question}.pl"])


# ----------------------------------------------------------------------------- parsing
_TIE_WORDS = (
    "tie", "no difference", "no noticeable difference", "equal", "equally", "the same", "same",
    "neither", "both", "cannot decide", "can't decide", "cannot tell", "can't tell", "draw",
    "bez różnicy", "bez roznicy", "remis", "takie same", "nie ma różnicy", "żadne", "oba",
)
# words that may follow a leading "A"/"B" verdict (otherwise a leading "A" may be an article)
_VERDICT_NEXT = {"is", "sounds", "matches", "better", "fits", "wins", "because", "since",
                 "jest", "brzmi", "lepiej", "pasuje", "bo", "ponieważ"}
_FIRST = r"(?:first|1st|audio\s*1|recording\s*1|clip\s*1|pierwsze)"
_SECOND = r"(?:second|2nd|audio\s*2|recording\s*2|clip\s*2|drugie)"


def parse_answer(text: str) -> str:
    """Robustly map a free-text judge answer to ``"A" | "B" | "tie"``.

    Order of rules: (1) the answer starts with A/B/TIE; (2) an explicit "answer: X" /
    "odpowiedź: X"; (3) tie phrases; (4) exactly one of A/B (or first/second) is mentioned;
    otherwise "tie" (unparseable answers count as no preference).
    """
    if not text:
        return "tie"
    s = text.strip().strip("*_`\"' .:\n")
    if s.lower() in ("a", "b"):
        return s.upper()
    m = re.match(r"^(?:(?i:recording|audio|nagranie|option|clip)\s+)?([AB])"
                 r"(?:$|[\s.,:;!)\]](?:\s*(\w+))?)", s)
    if m:
        nxt = (m.group(2) or "").lower()
        prefixed = m.group(0)[0] not in "AB"
        if prefixed or not nxt or nxt in _VERDICT_NEXT or not s[1:2].isspace():
            return m.group(1)
    if re.match(r"(?i)^(tie|remis|draw)\b", s):
        return "tie"
    m = re.search(r"(?i)(?:answer|odpowied[źz]|choice|wyb[óo]r|verdict|preference)\s*(?:is)?\s*[:\-]?\s*"
                  r"\**\s*(?:recording|audio|nagranie)?\s*((?-i:[AB])|tie|remis)\b", s)
    if m:
        g = m.group(1)
        return g.upper() if g.upper() in ("A", "B") else "tie"
    low = s.lower()
    if any(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", low) for w in _TIE_WORDS):
        return "tie"
    scan = re.sub(r"^A\s+(?=[a-z])", "", s)  # leading article "A ..."
    found = set(re.findall(r"(?<![A-Za-z])([AB])(?![A-Za-z])", scan))
    found |= {"A"} if re.search(rf"(?i)\b{_FIRST}\b", s) else set()
    found |= {"B"} if re.search(rf"(?i)\b{_SECOND}\b", s) else set()
    if len(found) == 1:
        return found.pop()
    return "tie"


# ----------------------------------------------------------------------------- judges
class Judge(Protocol):
    name: str
    questions: tuple[str, ...]

    def compare(self, path_a: Path, path_b: Path, text: str, question: str) -> str: ...


class FakeJudge:
    """Deterministic judge for tests: prefers the clip with the higher ``score_fn(path)``.

    ``position_bias="A"`` makes it always answer A (flip rate 1); ``tie_margin`` gives ties.
    """

    name = "fake"
    questions = QUESTIONS

    def __init__(self, score_fn: Callable[[Path], float], position_bias: str | None = None,
                 tie_margin: float = 0.0) -> None:
        self.score_fn, self.position_bias, self.tie_margin = score_fn, position_bias, tie_margin

    def compare(self, path_a: Path, path_b: Path, text: str, question: str) -> str:
        if self.position_bias:
            return self.position_bias
        d = self.score_fn(Path(path_a)) - self.score_fn(Path(path_b))
        return "tie" if abs(d) <= self.tie_margin else ("A" if d > 0 else "B")


class AudioboxAestheticsJudge:
    """Second, learned judge (§5.4): per-clip aesthetic scores -> pairwise choice by score.

    It does not read the text, so it answers only realism (PQ, production quality) and overall
    (CE, content enjoyment); "match" is skipped. Scores are cached per path.
    TODO(M1): verify package ``audiobox_aesthetics`` API (``initialize_predictor`` /
    ``predictor.forward([{"path": ...}])`` returning dicts with CE, CU, PC, PQ).
    """

    name = "audiobox-aesthetics"
    questions = ("realism", "overall")
    AXIS = {"realism": "PQ", "overall": "CE"}

    def __init__(self, tie_margin: float = 0.0, scorer: Callable[[Path], dict] | None = None):
        self.tie_margin = tie_margin
        self._scorer = scorer
        self._cache: dict[str, dict] = {}

    def _score(self, path: Path) -> dict:
        key = str(path)
        if key not in self._cache:
            if self._scorer is None:
                from audiobox_aesthetics.infer import initialize_predictor

                predictor = initialize_predictor()
                self._scorer = lambda p: predictor.forward([{"path": str(p)}])[0]
            self._cache[key] = self._scorer(Path(path))
        return self._cache[key]

    def compare(self, path_a: Path, path_b: Path, text: str, question: str) -> str:
        axis = self.AXIS[question]
        d = float(self._score(path_a)[axis]) - float(self._score(path_b)[axis])
        return "tie" if abs(d) <= self.tie_margin else ("A" if d > 0 else "B")


class QwenAudioJudge:
    """Qwen2-Audio-7B-Instruct pairwise judge (fixed revision, greedy decoding, optional 4-bit).

    Two clips per turn (A first, B second), loudness-normalised to −23 LUFS and resampled to the
    processor's rate (16 kHz, mono). TODO(M1): pin ``revision``; check that the installed
    transformers uses ``audio=`` (new) or ``audios=`` (old) in the processor call.
    """

    name = "qwen2-audio"
    questions = QUESTIONS

    def __init__(self, model_id: str = "Qwen/Qwen2-Audio-7B-Instruct", *, load_in_4bit: bool = False,
                 device_map: str = "auto", revision: str | None = None,
                 loudness_lufs: float = -23.0, max_new_tokens: int = 8) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        kw: dict = {"device_map": device_map, "revision": revision, "torch_dtype": torch.float16}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
                                                           bnb_4bit_compute_dtype=torch.float16)
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(model_id, **kw).eval()
        self.template = load_template()
        self.loudness_lufs, self.max_new_tokens = loudness_lufs, max_new_tokens
        self.sr = int(self.processor.feature_extractor.sampling_rate)
        self.last_raw: str = ""

    @lru_cache(maxsize=64)  # noqa: B019 (instance lives for the whole run)
    def _load(self, path: str):
        from audiocfg.audio_io import load_audio, normalize_loudness, resample

        audio, sr = load_audio(path)
        audio = normalize_loudness(audio, sr, self.loudness_lufs)
        return resample(audio.mean(axis=0, keepdims=True), sr, self.sr)[0]

    def compare(self, path_a: Path, path_b: Path, text: str, question: str) -> str:
        import torch

        conversation = [
            {"role": "system", "content": self.template["system"]},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": str(path_a)},
                {"type": "audio", "audio_url": str(path_b)},
                {"type": "text", "text": render_prompt(text, question, self.template)},
            ]},
        ]
        chat = self.processor.apply_chat_template(conversation, add_generation_prompt=True,
                                                  tokenize=False)
        audios = [self._load(str(path_a)), self._load(str(path_b))]
        inputs = self.processor(text=chat, audio=audios, sampling_rate=self.sr,
                                return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=False)
        out = out[:, inputs["input_ids"].shape[1]:]
        self.last_raw = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        return parse_answer(self.last_raw)


def make_judge(name: str, **kw) -> Judge:
    if name in ("qwen", "qwen2-audio"):
        return QwenAudioJudge(**kw)
    if name in ("audiobox", "audiobox-aesthetics"):
        return AudioboxAestheticsJudge(**kw)
    raise ValueError(f"unknown judge {name!r}")
