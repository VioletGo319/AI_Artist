"""Run on the private device after accepting the BFL model access terms.

Uses the locally configured HF token; never prints it or transfers it to HPC.
Already downloaded files are reused. No inference or Slurm submission here.
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

MODELS = (
    ("black-forest-labs/flux-3-image-klein", "3b767ff9b5261d71645e2ec49d87d298d955238d", "flux-3-image-klein",
     ["model.safetensors", "config.json", "manifest.json", "ae.safetensors", "text_encoder/*",
      "README.md", "LICENSE.md", "NOTICE.md"]),
    ("black-forest-labs/flux-3-image-klein-pup", "a57899549f8f6f64c6c4d88db30827d496ff76d0", "pup",
     ["*.json", "*.safetensors", "*.jinja", "tokenizer*", "*.txt", "*.model", "README.md", "LICENSE*"]),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("weights"))
    parser.add_argument("--check-access", action="store_true")
    args = parser.parse_args()
    api = HfApi()
    # A small authenticated access check precedes any large download.
    for repo, revision, folder, patterns in MODELS:
        api.auth_check(repo_id=repo, repo_type="model")
        print(f"Access OK: {repo}")
    if args.check_access:
        return
    for repo, revision, folder, patterns in MODELS:
        snapshot_download(repo, revision=revision, local_dir=args.weights / folder, allow_patterns=patterns)
    print("Pinned model caches downloaded. Transfer only weights, excluding .cache; never transfer HF/OAuth tokens.")


if __name__ == "__main__":
    main()
