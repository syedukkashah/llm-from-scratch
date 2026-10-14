"""Builds the folder to upload to a Hugging Face Space (Gradio SDK).

    python deploy/huggingface/make_space.py

Creates deploy/huggingface/space/ with README.md, app.py, requirements.txt, src/ and configs/.
Upload everything inside it to the Space (Files > Add file > Upload files).
"""
import shutil
from pathlib import Path

here = Path(__file__).parent
root = here.parent.parent
out = here / "space"

if out.exists():
    shutil.rmtree(out)
out.mkdir()

for name in ("README.md", "app.py", "requirements.txt"):
    shutil.copy(here / name, out / name)
shutil.copytree(root / "src", out / "src", ignore=shutil.ignore_patterns("__pycache__"))
shutil.copytree(root / "configs", out / "configs")

for p in sorted(out.rglob("*")):
    if p.is_file():
        print(p.relative_to(out))
