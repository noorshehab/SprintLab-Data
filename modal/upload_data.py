"""One-time upload of the local sprintlabfiles/ proxy dataset into the
`noxed-data` Modal Volume, so notebooks executed on Modal read the same
files a local run would read from ./data.

Usage:
    python modal/upload_data.py /path/to/sprintlabfiles
"""
import sys
from pathlib import Path

import modal

VOLUME_NAME = "noxed-data"
FILES = [
    "analysis_metadata.csv", "chronological_delta.csv", "kc_metadata.csv",
    "kc_tree_structure.md", "practice_effect_perKC.csv", "practice_effect_perQ.csv",
    "question_diffrentiation.csv", "question_diffrentiation_cumulative.csv",
    "question_metadata.csv", "responses.csv", "tree_translation.txt",
]


def main(local_dir: str):
    src = Path(local_dir)
    missing = [f for f in FILES if not (src / f).exists()]
    if missing:
        raise SystemExit(f"Missing expected data files in {src}: {missing}")

    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    with volume.batch_upload(force=True) as batch:
        for fname in FILES:
            print(f"uploading {fname} ...")
            batch.put_file(str(src / fname), f"/{fname}")
    print(f"Done. Volume '{VOLUME_NAME}' now holds {len(FILES)} files.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "./data")
