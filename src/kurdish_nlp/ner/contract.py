"""Studio-compatible entities payload, built exclusively from actual spaCy output.

No web runtime dependency and no model loading or routing changes. The API's
existing adapter already exposes this shape. Offsets refer to original input,
in Python Unicode code points, with an exclusive end.
"""


def entities_payload(doc, model_id: str, version: str, labels: list[str]) -> dict:
    entities = []
    for entity in doc.ents:
        start, end = entity.start_char, entity.end_char
        if not 0 <= start < end <= len(doc.text) or doc.text[start:end] != entity.text:
            raise ValueError("Entity offsets do not match original text")
        if entity.label_ not in labels:
            raise ValueError("Model emitted a label outside its declared inventory")
        entities.append(
            {"text": entity.text, "label": entity.label_, "start": start, "end": end}
        )
    return {
        "model": {"id": model_id, "name": "Sorani NER", "version": version},
        "labels": sorted(labels),
        "entities": entities,
    }
