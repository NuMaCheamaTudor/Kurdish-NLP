"""Task routing: Language ID always; NER and grammar only through real models."""

from __future__ import annotations

from pathlib import Path

from app.errors import ApiError
from app.languages import language_info
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse, LanguageResolution
from app.schemas.detect import DetectRequest, DetectResponse
from app.schemas.models import ModelInfo, TaskInfo
from app.services.langid import LanguageIdService
from app.services.spacy_models import SpacyTaskModel, TaskModelSpec

TASK_NAMES = {"language": "Language", "entities": "Named entities", "grammar": "Grammar"}
TASK_UNSUPPORTED_LANGUAGE = {
    "entities": "Named entity recognition is not available for {language}.",
    "grammar": "Grammar analysis is not available for {language}.",
}
TASK_UNAVAILABLE = {
    "entities": "NER model is not installed.",
    "grammar": "Grammar model is not installed.",
}


def legacy_task_specs(ner_path: Path | None, grammar_path: Path | None) -> list[TaskModelSpec]:
    return [
        TaskModelSpec(
            id="kurdish-ner-v1",
            task="entities",
            name="Sorani NER",
            description="spaCy pipeline with an XLM-RoBERTa encoder for Sorani named entities.",
            languages=("ckb",),
            path=ner_path,
            required_components=(("ner",),),
        ),
        TaskModelSpec(
            id="kurdish-pos-v1",
            task="grammar",
            name="Kurmanji grammar",
            description=(
                "spaCy pipeline with an XLM-RoBERTa encoder for Kurmanji POS tags, "
                "morphology and dependencies."
            ),
            languages=("kmr",),
            path=grammar_path,
            required_components=(("tagger", "morphologizer"), ("parser",)),
        ),
    ]


class AnalysisService:
    def __init__(
        self,
        langid: LanguageIdService,
        task_specs: list[TaskModelSpec],
        max_text_chars: int,
    ) -> None:
        self.langid = langid
        self.task_models = [SpacyTaskModel(spec) for spec in task_specs]
        self.max_text_chars = max_text_chars

    def load(self) -> None:
        self.langid.load()
        for model in self.task_models:
            model.load()

    # Discovery ---------------------------------------------------------------

    def tasks(self) -> list[TaskInfo]:
        language_ready = self.langid.any_ready
        infos = [
            TaskInfo(
                id="language",
                name=TASK_NAMES["language"],
                available=language_ready,
                message=None if language_ready else "No language ID model is available.",
                languages=[] if not language_ready else list(_langid_languages(self.langid)),
            )
        ]
        for task in ("entities", "grammar"):
            ready = [m for m in self.task_models if m.spec.task == task and m.ready]
            infos.append(
                TaskInfo(
                    id=task,
                    name=TASK_NAMES[task],
                    available=bool(ready),
                    message=None if ready else TASK_UNAVAILABLE[task],
                    languages=sorted({code for m in ready for code in m.spec.languages}),
                )
            )
        return infos

    def task_model_infos(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                id=model.spec.id,
                name=model.spec.name,
                full_name=model.spec.name,
                task=model.spec.task,
                description=model.spec.description,
                languages=list(model.spec.languages),
                status="ready" if model.ready else "unavailable",
                default=False,
                unavailable_reasons=list(model.reasons),
                version=model.version,
                framework="spaCy",
            )
            for model in self.task_models
        ]

    # Requests ----------------------------------------------------------------

    def validate_text(self, text: str) -> None:
        if not text.strip():
            raise ApiError(422, "empty_text", "Enter some text to analyze.")
        if len(text) > self.max_text_chars:
            raise ApiError(
                422,
                "text_too_long",
                f"Text is longer than {self.max_text_chars:,} characters.",
                {"max_text_chars": self.max_text_chars, "length": len(text)},
            )

    def detect(self, request: DetectRequest) -> DetectResponse:
        self.validate_text(request.text)
        model = self.langid.resolve(request.model)
        return self.langid.detect(request.text, model, request.policy, request.top_k)

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        self.validate_text(request.text)
        # An unknown classifier id is always an error, even when a manual
        # language means the classifier will not run.
        self.langid.check_known(request.model)
        needs_detection = request.task == "language" or request.language == "auto"

        if request.task != "language":
            candidates = [m for m in self.task_models if m.spec.task == request.task and m.ready]
            if not candidates:
                raise ApiError(
                    503,
                    "task_unavailable",
                    TASK_UNAVAILABLE[request.task],
                    {"task": request.task},
                )

        detection: DetectResponse | None = None
        if needs_detection:
            model = self.langid.resolve(request.model)
            detection = self.langid.detect(request.text, model, request.policy, 7)

        if request.language == "auto":
            assert detection is not None
            routed = detection.language.code
        else:
            routed = request.language
        info = language_info(routed)
        resolution = LanguageResolution(
            code=info.code,
            name=info.name,
            alt_name=info.alt_name,
            mode="auto" if request.language == "auto" else "manual",
        )
        response = AnalyzeResponse(task=request.task, language=resolution, detection=detection)
        if request.task == "language":
            return response

        if routed == "und":
            raise ApiError(
                422,
                "language_undetermined",
                "The language could not be determined. Choose it manually.",
                {"task": request.task, "reason": detection.reason if detection else None},
            )
        task_model = next((m for m in candidates if routed in m.spec.languages), None)
        if task_model is None:
            supported = sorted({c for m in candidates for c in m.spec.languages})
            raise ApiError(
                422,
                "unsupported_language",
                TASK_UNSUPPORTED_LANGUAGE[request.task].format(language=info.name),
                {"task": request.task, "language": routed, "supported": supported},
            )
        result = task_model.run(request.text)
        if request.task == "entities":
            response.entities = result  # type: ignore[assignment]
        else:
            response.grammar = result  # type: ignore[assignment]
        return response


def _langid_languages(langid: LanguageIdService) -> list[str]:
    ready = [info for info in langid.describe() if info.status == "ready"]
    return ready[0].languages if ready else []
