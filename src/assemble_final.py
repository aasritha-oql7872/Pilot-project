"""Assemble the final, versioned submission ZIP from the package the Colab pipeline produced.

    python -m src.assemble_final --package-zip searchlite_rl_submission_v2_20261109_0200.zip \
        --deck SearchLite_RL_Final_Presentation.pptx --script SearchLite_RL_Speaking_Script.md \
        --notebooks SearchLite_RL_Final_Submission.ipynb SearchLite_RL_Final_Study.ipynb ... --version v3_20261108 --out-dir final_submission

Starts from the Colab package (report, figures, tables, raw evaluations, code, frozen adapter) and ADDS what Colab could not: the
presentation, the speaking script, the executed notebooks and a PDF of the report. It never edits an existing file of the package:
the Colab manifest is kept as MANIFEST_SHA256.colab.txt and a new MANIFEST_SHA256.txt covers every file. The output name carries the
version; an existing archive is never overwritten. Notebooks and everything else are scanned for secrets before anything is written.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from src.submission_check import FORBIDDEN, MAX_BYTES, SECRET

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
REQUIRED = ["FINAL_PILOT_REPORT.md", "README_SUBMISSION.md", "FROZEN_HARD.json", "case_studies.md", "tables/two_comparisons.md",
            "checkpoint/frozen_headline_adapter/adapter_model.safetensors", "code/src/train_grpo.py"]
CSS = """body{font-family:-apple-system,Helvetica,Arial,sans-serif;font-size:10.5pt;line-height:1.38;color:#1d1926;max-width:880px;margin:0 auto}
h1{color:#4e2a84;font-size:21pt;border-bottom:2px solid #4e2a84;padding-bottom:4px}h2{color:#4e2a84;font-size:15pt;margin-top:22px;border-bottom:1px solid #d9d2e3;padding-bottom:2px}
h3{color:#664396;font-size:12pt}table{border-collapse:collapse;font-size:8.6pt;margin:8px 0;width:100%}th,td{border:1px solid #cfc6dc;padding:3px 5px;vertical-align:top}
th{background:#f1e8fa}code{font-size:8.8pt;background:#f6f3fa;padding:0 2px}pre{background:#f6f3fa;padding:8px;font-size:8.4pt;white-space:pre-wrap}
img{max-width:100%;height:auto}blockquote{border-left:3px solid #8d6fb8;margin:8px 0;padding:2px 10px;background:#faf8fd}
@page{size:Letter;margin:16mm 14mm}"""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_extract(zpath, dest):
    dest = Path(dest).resolve()
    with zipfile.ZipFile(zpath) as z:
        bad = z.testzip()
        if bad:
            raise SystemExit(f"corrupt entry in {zpath}: {bad}")
        for name in z.namelist():
            if not (dest / name).resolve().is_relative_to(dest):
                raise SystemExit(f"unsafe path in zip: {name}")
        z.extractall(dest)


def md_to_pdf(md_path, pdf_path, chrome=CHROME):
    """Markdown report -> HTML -> PDF with headless Chrome. Returns (ok, reason). Relative figure paths resolve against the report's folder."""
    try:
        import markdown
    except ImportError:
        return False, "python 'markdown' package not installed"
    if not Path(chrome).exists():
        return False, f"Chrome not found at {chrome}"
    md_path, pdf_path = Path(md_path), Path(pdf_path)
    html = markdown.markdown(md_path.read_text(), extensions=["tables", "fenced_code", "sane_lists"])
    page = md_path.parent / ".report_render.html"
    page.write_text(f"<!doctype html><meta charset='utf-8'><title>SearchLite-RL final report</title><style>{CSS}</style>{html}")
    try:
        r = subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}", page.as_uri()],
                           capture_output=True, text=True, timeout=180)
    finally:
        page.unlink(missing_ok=True)
    return (pdf_path.exists() and pdf_path.stat().st_size > 10_000), (r.stderr.strip().splitlines() or [""])[-1]


def scan_secrets(root):
    hits = []
    for p in Path(root).rglob("*"):
        if p.is_file() and p.stat().st_size < 60 * 1024 * 1024 and p.suffix.lower() not in (".safetensors", ".png", ".pdf", ".pptx", ".zip"):
            m = SECRET.search(p.read_text(errors="ignore"))
            if m:
                hits.append(f"{p.relative_to(root)}: {m.group(0)[:16]}...")
    return hits


def hygiene(root):
    root = Path(root)
    bad = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if p.suffix.lower() in FORBIDDEN or p.name.startswith(".env") or "__pycache__" in p.parts or p.suffix == ".pyc":
            bad.append(f"forbidden: {rel}")
        if p.stat().st_size > MAX_BYTES and p.suffix != ".safetensors":
            bad.append(f"too large: {rel}")
    safes = [p.relative_to(root).as_posix() for p in root.rglob("*.safetensors")]
    if safes != ["checkpoint/frozen_headline_adapter/adapter_model.safetensors"]:
        bad.append(f"unexpected checkpoints: {safes}")
    return bad


def write_manifest(root):
    root = Path(root)
    lines = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "MANIFEST_SHA256.txt":
            lines.append(f"{sha256(p)}  {p.relative_to(root).as_posix()}")
    (root / "MANIFEST_SHA256.txt").write_text("\n".join(lines) + "\n")
    return len(lines)


def verify_manifest(root):
    root = Path(root)
    bad = []
    for line in (root / "MANIFEST_SHA256.txt").read_text().splitlines():
        h, _, rel = line.partition("  ")
        if not (root / rel).exists() or sha256(root / rel) != h:
            bad.append(rel)
    return bad


def assemble(package_zip, deck, script, notebooks, version, out_dir, deck_pdf=None, chrome=CHROME, build_pdf=True):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"searchlite_rl_submission_{version}.zip"
    if target.exists():
        raise SystemExit(f"{target} already exists; choose a new --version (an archived submission is never overwritten)")
    report = {"source_zip": str(package_zip), "source_zip_sha256": sha256(package_zip), "added": [], "skipped": []}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "pkg"; root.mkdir()
        safe_extract(package_zip, root)
        missing = [r for r in REQUIRED if not (root / r).exists()]
        if missing:
            raise SystemExit(f"source package is missing: {missing}")
        if (root / "MANIFEST_SHA256.txt").exists():
            shutil.move(root / "MANIFEST_SHA256.txt", root / "MANIFEST_SHA256.colab.txt")
        pres = root / "presentation"; pres.mkdir(exist_ok=True)
        for src, name in ((deck, "SearchLite_RL_Final_Presentation.pptx"), (deck_pdf, "SearchLite_RL_Final_Presentation.pdf"), (script, "SearchLite_RL_Speaking_Script.md")):
            if src and Path(src).exists():
                shutil.copy(src, pres / name); report["added"].append(f"presentation/{name}")
            elif src:
                report["skipped"].append(f"presentation/{name}: {src} not found")
        nbdir = root / "notebooks"; nbdir.mkdir(exist_ok=True)
        for nb in notebooks:
            nb = Path(nb)
            if not nb.exists():
                report["skipped"].append(f"{nb.name}: not found"); continue
            if SECRET.search(nb.read_text(errors="ignore")):
                report["skipped"].append(f"{nb.name}: contains a secret-like string; NOT packaged"); continue
            if (nbdir / nb.name).exists():
                report["skipped"].append(f"{nb.name}: a notebook with this name is already in the package; not overwritten"); continue
            shutil.copy(nb, nbdir / nb.name); report["added"].append(f"notebooks/{nb.name}")
        if build_pdf:
            ok, why = md_to_pdf(root / "FINAL_PILOT_REPORT.md", root / "FINAL_PILOT_REPORT.pdf", chrome)
            (report["added"] if ok else report["skipped"]).append("FINAL_PILOT_REPORT.pdf" + ("" if ok else f": {why}"))
        sc = (root / "STATUS_CHECKLIST.md").read_text() if (root / "STATUS_CHECKLIST.md").exists() else ""
        inherited = [l for l in sc.splitlines() if "| FAIL |" in l]
        (root / "ASSEMBLY_NOTES.md").write_text(
            "# Assembly notes (added after the Colab pipeline)\n\n"
            f"This archive was assembled from the Colab package (source ZIP sha256 `{report['source_zip_sha256']}`) by `src/assemble_final.py`. "
            "No file of the Colab package was edited: its manifest is kept as `MANIFEST_SHA256.colab.txt`, `STATUS_CHECKLIST.md` is the unmodified Colab checklist.\n\n"
            "Added here: " + ", ".join(f"`{a}`" for a in report["added"]) + ".\n\n"
            + ("Not added: " + "; ".join(report["skipped"]) + ".\n\n" if report["skipped"] else "")
            + "FAIL rows in the Colab checklist, quoted unmodified:\n\n" + ("\n".join(inherited) if inherited else "(none)") + "\n\n"
            "The executed-notebook row is resolved by `notebooks/`. If a test-suite FAIL is listed above, see `tables/test_output.txt` if present; the "
            "failure is reported, not hidden.\n")
        problems = hygiene(root) + [f"secret-like string: {h}" for h in scan_secrets(root)]
        if problems:
            raise SystemExit("package hygiene problems, nothing written:\n  " + "\n  ".join(problems))
        n = write_manifest(root)
        bad = verify_manifest(root)
        if bad:
            raise SystemExit(f"manifest verification failed: {bad}")
        with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(root).as_posix())
        report.update({"zip": str(target), "zip_sha256": sha256(target), "zip_bytes": target.stat().st_size, "manifest_files": n})
    (out_dir / f"{target.name}.sha256").write_text(f"{report['zip_sha256']}  {target.name}\n")
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        report["entries"] = len(z.namelist())
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package-zip", required=True)
    ap.add_argument("--deck", default=None)
    ap.add_argument("--deck-pdf", default=None)
    ap.add_argument("--script", default=None)
    ap.add_argument("--notebooks", nargs="*", default=[])
    ap.add_argument("--version", required=True)
    ap.add_argument("--out-dir", default="final_submission")
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args()
    r = assemble(a.package_zip, a.deck, a.script, a.notebooks, a.version, a.out_dir, a.deck_pdf, build_pdf=not a.no_pdf)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
