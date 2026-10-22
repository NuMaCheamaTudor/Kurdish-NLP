"""Prepare the code-only transfer capsule for the checked-in Colab notebook.

Does not copy/upload source data, download models, commit, or run training.
"""

import argparse
import json
from pathlib import Path

from kurdish_nlp.ner.colab_artifacts import prepare_capsule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ner/colab-stress-fix/sorani-ner-v2-code.zip"),
    )
    args = parser.parse_args()
    result = prepare_capsule(Path.cwd(), args.output)
    args.output.with_suffix(".sha256").write_text(result["archive_sha256"] + "\n")
    notebook = Path("notebooks/sorani_ner_v2_colab.ipynb")
    if notebook.is_file():
        content = json.loads(notebook.read_text())
        identity = next(
            c
            for c in content["cells"]
            if c.get("metadata", {}).get("role") == "transfer_identity"
        )
        identity["source"] = [
            f"EXPECTED_CODE_SHA256 = {json.dumps(result['archive_sha256'])}\n",
            f"EXPECTED_BASE_COMMIT = {json.dumps(result['base_git_commit'])}\n",
            f"EXPECTED_CONTENT_SHA256 = {json.dumps(result['content_sha256'])}",
        ]
        notebook.write_text(json.dumps(content, ensure_ascii=False, indent=1) + "\n")
        # Keep the user-facing upload copy byte-identical to the canonical source.
        Path("sorani_ner_v2_colab.ipynb").write_bytes(notebook.read_bytes())
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "archive",
                    "archive_sha256",
                    "base_git_commit",
                    "content_sha256",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
