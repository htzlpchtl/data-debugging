"""Historical download utility. Not needed to inspect this checkpoint."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
from huggingface_hub import snapshot_download

repo = "mlx-community/Qwen2.5-7B-Instruct-4bit"
revision = "c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed"
path = snapshot_download(repo, revision=revision, allow_patterns=[
    "*.json", "*.safetensors", "*.txt", "*.model", "README.md"
], max_workers=2)
manifest = {"repo": repo, "revision": revision,
            "local_path": os.path.relpath(path, ROOT)}
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results/model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
