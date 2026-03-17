import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

__all__ = ["BatchJob", "BatchParseResult", "BatchLinker"]


@dataclass(frozen=True)
class BatchJob:
    sample: Path
    reference: Path | None = None


@dataclass
class BatchParseResult:
    jobs: list[BatchJob]
    reference_only: list[Path]
    errors: list[str]


class BatchLinker:
    """Match sample and reference files."""

    _SAMPLE_PATTERNS = re.compile(r"(sample|samp|sam)\d*")
    _REF_PATTERNS = re.compile(r"(reference|ref)\d*")
    _SPLIT_PATTERN = re.compile(r"[^a-z0-9]+")
    _BLOCKED_TOKENS = {"sample", "samp", "sam", "ref", "reference"}

    def __init__(self, files: Iterable[Path], similarity_threshold: float = 0.82):
        selected = [Path(p).expanduser().resolve() for p in files]
        self.files = sorted(selected, key=lambda p: p.name.lower())
        self.similarity_threshold = float(similarity_threshold)

    @classmethod
    def _tokenize_stem(cls, stem: str) -> list[str]:
        lowered = str(stem).lower()
        lowered = cls._SAMPLE_PATTERNS.sub(" sample ", lowered)
        lowered = cls._REF_PATTERNS.sub(" ref ", lowered)
        return [tok for tok in cls._SPLIT_PATTERN.split(lowered) if tok]

    @classmethod
    def _is_reference_name(cls, stem: str) -> bool:
        return "ref" in cls._tokenize_stem(stem)

    @classmethod
    def _is_sample_name(cls, stem: str) -> bool:
        tokens = cls._tokenize_stem(stem)
        return "sample" in tokens or "samp" in tokens or "sam" in tokens

    @classmethod
    def _derive_pair_key(cls, stem: str) -> str:
        tokens = [
            tok for tok in cls._tokenize_stem(stem)
            if tok not in cls._BLOCKED_TOKENS and not tok.isdigit()
        ]
        return "_".join(tokens) if tokens else str(stem).lower()

    def auto_link(self) -> tuple[list[BatchJob], list[Path]]:
        """Pair sample files with the closest reference file."""
        sample_files: list[Path] = []
        reference_files: list[Path] = []
        for path in self.files:
            stem = path.stem
            if self._is_reference_name(stem) and not self._is_sample_name(stem):
                reference_files.append(path)
            else:
                sample_files.append(path)

        refs_by_key: dict[str, list[Path]] = {}
        for ref_path in reference_files:
            refs_by_key.setdefault(self._derive_pair_key(ref_path.stem), []).append(ref_path)

        ref_keys = list(refs_by_key.keys())
        jobs: list[BatchJob] = []
        matched_refs: set[Path] = set()

        for sample_path in sample_files:
            sample_key = self._derive_pair_key(sample_path.stem)
            match = None
            if sample_key in refs_by_key:
                match = refs_by_key[sample_key][0]
            elif ref_keys:
                best_key = None
                best_score = 0.0
                for key in ref_keys:
                    score = SequenceMatcher(None, sample_key, key).ratio()
                    if score > best_score:
                        best_score = score
                        best_key = key
                if best_key is not None and best_score >= self.similarity_threshold:
                    match = refs_by_key[best_key][0]
            jobs.append(BatchJob(sample=sample_path, reference=match))
            if match is not None:
                matched_refs.add(match)

        reference_only = [path for path in reference_files if path not in matched_refs]
        return jobs, reference_only

    @staticmethod
    def render_mapping_text(jobs: list[BatchJob], reference_only: list[Path]) -> str:
        """Return editable mapping lines."""
        lines = ["# One job per line: sample_path ==> reference_path(optional)"]
        for job in jobs:
            right = str(job.reference) if job.reference is not None else ""
            lines.append(f"{job.sample} ==> {right}")
        if reference_only:
            lines.append("# Reference-only lines (not executed, editable)")
            for ref_path in reference_only:
                lines.append(f"==> {ref_path}")
        return "\n".join(lines)

    @staticmethod
    def _resolve_path(raw_value: str, pool: list[Path]) -> Path | None:
        token = str(raw_value or "").strip().strip("`").strip('"').strip("'")
        if not token:
            return None

        candidate = Path(token).expanduser()
        if candidate.exists():
            return candidate.resolve()
        if not candidate.is_absolute():
            cwd_candidate = (Path.cwd() / candidate).expanduser()
            if cwd_candidate.exists():
                return cwd_candidate.resolve()

        token_norm = token.lower()
        exact = [p for p in pool if str(p).lower() == token_norm]
        if len(exact) == 1:
            return exact[0]
        by_name = [p for p in pool if p.name.lower() == Path(token).name.lower()]
        if len(by_name) == 1:
            return by_name[0]
        return None

    @staticmethod
    def _split_mapping_line(line: str) -> tuple[str, str]:
        if "==>" in line:
            return tuple((line.split("==>", 1) + [""])[:2])
        if "|" in line:
            return tuple((line.split("|", 1) + [""])[:2])
        return line, ""

    @classmethod
    def parse_mapping_text(cls, text: str, pool: Iterable[Path]) -> BatchParseResult:
        """Parse editable mapping lines."""
        pool_paths = [Path(p).expanduser().resolve() for p in pool]
        jobs: list[BatchJob] = []
        reference_only: list[Path] = []
        errors: list[str] = []

        for line_no, raw_line in enumerate(str(text or "").splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            left_raw, right_raw = cls._split_mapping_line(line)
            sample_raw = left_raw.strip()
            reference_raw = right_raw.strip()

            sample_path = cls._resolve_path(sample_raw, pool_paths)
            reference_path = cls._resolve_path(reference_raw, pool_paths)

            if sample_raw and sample_path is None:
                errors.append(f"Line {line_no}: sample not found -> {sample_raw}")
            if reference_raw and reference_path is None:
                errors.append(f"Line {line_no}: reference not found -> {reference_raw}")

            if sample_path is not None:
                jobs.append(BatchJob(sample=sample_path, reference=reference_path))
            elif reference_path is not None:
                reference_only.append(reference_path)

        deduped_jobs: list[BatchJob] = []
        seen: set[tuple[str, str]] = set()
        for job in jobs:
            key = (str(job.sample), str(job.reference) if job.reference is not None else "")
            if key in seen:
                continue
            seen.add(key)
            deduped_jobs.append(job)

        return BatchParseResult(
            jobs=deduped_jobs,
            reference_only=reference_only,
            errors=errors,
        )
