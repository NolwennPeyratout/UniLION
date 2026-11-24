"""Check that a BEV feature folder covers every sample of a LLaVA-style json.

Each json entry looks like
    {"image": [".../<token>.pt"], "history": {"token": "<token>"}, ...}
and DrivePI loads ``<feature_dir>/<token><ext>``.

Usage:
    python tools/check_bev_features.py \
        /path/LiDAR-LLM-Nu-Caption/train_llava.json \
        /path/nuscenes/unilion_bev_feats_train [--ext .npz]
"""
import argparse
import json
import os
import sys
from pathlib import Path


def expected_tokens(llava_json):
    with open(llava_json, "r", encoding="utf-8") as file:
        entries = json.load(file)

    tokens = []
    inconsistent = []
    for index, entry in enumerate(entries):
        token = entry.get("history", {}).get("token")
        image_tokens = [Path(path).stem for path in entry.get("image", [])]
        if token is None:
            if not image_tokens:
                raise ValueError(f"{llava_json}: entry {index} has no token")
            token = image_tokens[0]
        if any(image_token != token for image_token in image_tokens):
            inconsistent.append((token, image_tokens))
        tokens.append(token)
    return tokens, inconsistent


def check_bev_features(llava_json, feature_dir, ext=".npz", verbose=True):
    """Return the sorted list of tokens of ``llava_json`` without a
    ``<token><ext>`` file in ``feature_dir``."""
    tokens, inconsistent = expected_tokens(llava_json)
    unique_tokens = set(tokens)
    present = {
        name[: -len(ext)]
        for name in os.listdir(feature_dir)
        if name.endswith(ext)
    }
    missing = sorted(unique_tokens - present)

    if verbose:
        print(f"{llava_json}")
        print(f"  entries: {len(tokens)} ({len(unique_tokens)} unique tokens)")
        print(f"  '{ext}' files in {feature_dir}: {len(present)}")
        print(f"  missing: {len(missing)}")
        for token in missing[:20]:
            print(f"    {token}")
        if len(missing) > 20:
            print(f"    ... ({len(missing) - 20} more)")
        if inconsistent:
            print(
                f"  WARNING: {len(inconsistent)} entries whose image file name "
                f"differs from history.token, e.g. {inconsistent[0]}"
            )
    return missing


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("llava_json", help="train_llava.json or val_llava.json")
    parser.add_argument("feature_dir", help="Folder with <token><ext> features")
    parser.add_argument("--ext", default=".npz", help="Feature file extension")
    args = parser.parse_args()
    missing = check_bev_features(args.llava_json, args.feature_dir, args.ext)
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
