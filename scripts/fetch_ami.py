#!/usr/bin/env python3
"""Fetch the AMI Meeting Corpus (CC BY 4.0).

    python scripts/fetch_ami.py --out data/raw/ami --meetings ES2002a ES2002b

Annotations are one 23 MB zip covering all 171 meetings. Audio is per meeting
(~40 MB each for the mixed headset track), so only the meetings you ask for are
downloaded. Existing files are left alone, so re-running resumes the set.

Corpus: https://groups.inf.ed.ac.uk/ami/corpus/  -- please cite it if you publish.
"""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

ANNOTATIONS_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
AUDIO_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{meeting}/audio/{meeting}.{track}.wav"

DEFAULT_MEETINGS = ["ES2002a", "ES2002b", "ES2002c", "ES2002d"]


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  get  {dest.name} ...", end="", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    print(f" {dest.stat().st_size / 1e6:.1f} MB")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/raw/ami")
    parser.add_argument("--meetings", nargs="*", default=DEFAULT_MEETINGS)
    parser.add_argument("--track", default="Mix-Headset")
    parser.add_argument("--annotations-only", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    print("annotations:")
    zip_path = download(ANNOTATIONS_URL, out / "ami_public_manual_1.6.2.zip")
    extracted = out / "ami_public_manual_1.6.2"
    if not (extracted / "words").is_dir():
        print(f"  unzip -> {extracted}")
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extracted)

    if not args.annotations_only:
        print(f"audio ({args.track}):")
        for meeting in args.meetings:
            url = AUDIO_URL.format(meeting=meeting, track=args.track)
            download(url, out / "amicorpus" / meeting / "audio" / f"{meeting}.{args.track}.wav")

    print(f"\nannotations: {extracted}")
    print(f"audio root : {out / 'amicorpus'}")
    print("AMI is CC BY 4.0 -- attribute the corpus in any published work.")


if __name__ == "__main__":
    main()
