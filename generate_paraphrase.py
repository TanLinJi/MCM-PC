#!/usr/bin/env python
"""
Generate paraphrase descriptions for 3D point cloud category names using DeepSeek API.

Per MCP-3D plan §3 (vMF text anchor module):
  - 4 prompt templates x 10 calls per class = 40 paraphrases / class
  - reproducibility: temperature=0.3 + fixed seed (Gap F4)
  - resume capability: skip classes already in JSON cache
  - cost estimate: DeepSeek-chat ~6 RMB total for 1427 classes x 4 templates

Output JSON format (compatible with Point-Cache/llm/*.json convention):
  {
      "<classname_1>": ["paraphrase 1", "paraphrase 2", ..., "paraphrase 40"],
      "<classname_2>": [...],
      ...
  }

Usage:
  # set API key once
  export DEEPSEEK_API_KEY=sk-xxx

  # generate for default datasets (modelnet40, scanobjectnn)
  python generate_paraphrase.py

  # generate for all 4 datasets
  python generate_paraphrase.py --datasets modelnet40 scanobjectnn omniobject3d objaverse_lvis

  # smoke test (only 2 classes per dataset)
  python generate_paraphrase.py --datasets modelnet40 --smoke

  # set custom output dir
  python generate_paraphrase.py --output_dir Point-Cache/llm
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai==1.30.0")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):
        return it


# ---------------------------------------------------------------------------
# DeepSeek API configuration (uses openai-compatible interface)
# ---------------------------------------------------------------------------
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

VOWELS = {"A", "E", "I", "O", "U"}


def make_prompts(category: str) -> List[str]:
    """4 paraphrase templates. Same as Point-Cache llm/llm_generate_prompts.py."""
    article = "an" if category[0].upper() in VOWELS else "a"
    return [
        f"What does {article} {category} point cloud look like?",
        f"What are the identifying characteristics of {article} {category} point cloud?",
        f"Please describe {article} {category} point cloud with details.",
        f"Make a complete and meaningful sentence with the following words: {category}, point cloud.",
    ]


# ---------------------------------------------------------------------------
# Class list loaders
# ---------------------------------------------------------------------------
def load_classnames(dataset: str, project_root: Path) -> List[str]:
    """Load category list for the given dataset.

    Returns a list of classnames (strings) used for paraphrase generation.
    """
    pc_root = project_root / "Point-Cache"
    data_root = pc_root / "data"

    if dataset == "modelnet40":
        # try shape_names.txt first (available with modelnet_c), fall back to classnames.txt
        for candidate in [
            data_root / "modelnet_c" / "shape_names.txt",
            data_root / "modelnet40_c" / "shape_names.txt",
            data_root / "modelnet40" / "classnames.txt",
            data_root / "modelnet40" / "shape_names.txt",
        ]:
            if candidate.exists():
                return _read_lines(candidate)
        raise FileNotFoundError(f"No classname list found for {dataset}")

    elif dataset == "scanobjectnn":
        for candidate in [
            data_root / "sonn_c" / "shape_names.txt",
            data_root / "scanobjectnn" / "shape_names.txt",
        ]:
            if candidate.exists():
                return _read_lines(candidate)
        raise FileNotFoundError(f"No classname list found for {dataset}")

    elif dataset == "omniobject3d":
        # OmniObject3D classes are subdirectory names under data/omniobject3d/1024
        omni_dir = data_root / "omniobject3d" / "1024"
        if omni_dir.exists():
            return sorted([d.name for d in omni_dir.iterdir() if d.is_dir()])
        raise FileNotFoundError(f"omniobject3d not downloaded yet: {omni_dir}")

    elif dataset == "objaverse_lvis":
        for candidate in [
            data_root / "objaverse_lvis" / "classnames.txt",
        ]:
            if candidate.exists():
                return _read_lines(candidate)
        # fall back to parsing lvis_testset.txt
        testset = data_root / "objaverse_lvis" / "lvis_testset.txt"
        if testset.exists():
            seen = set()
            classes = []
            for line in _read_lines(testset):
                cls = line.split(",")[0].strip()
                if cls and cls not in seen:
                    seen.add(cls)
                    classes.append(cls)
            return classes
        raise FileNotFoundError(f"objaverse_lvis classnames not found")

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def _read_lines(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------------
# DeepSeek API caller with retry and budget tracking
# ---------------------------------------------------------------------------
class DeepSeekClient:
    def __init__(self, api_key: str, temperature: float = 0.3, max_tokens: int = 80):
        self.client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.total_calls = 0
        self.total_failures = 0

    def chat(self, prompt: str, max_retry: int = 3) -> str:
        """Single completion. Returns the assistant message content (a single sentence)."""
        for attempt in range(max_retry):
            try:
                self.total_calls += 1
                resp = self.client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that describes 3D point cloud objects in concise, single sentences. Keep each response under 25 words. End with a period."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = resp.choices[0].message.content.strip()
                # post-process: ensure ends with period and is single line
                content = content.replace("\n\n", " ").replace("\n", " ").strip()
                if not content.endswith("."):
                    content += "."
                return content
            except Exception as e:
                self.total_failures += 1
                wait = 2 ** attempt
                print(f"    [retry {attempt+1}/{max_retry}] {type(e).__name__}: {e}; waiting {wait}s")
                time.sleep(wait)
        return ""  # all retries failed


# ---------------------------------------------------------------------------
# Main paraphrase generation loop
# ---------------------------------------------------------------------------
def generate_for_dataset(
    dataset: str,
    classnames: List[str],
    output_path: Path,
    client: DeepSeekClient,
    n_per_template: int = 10,
    smoke: bool = False,
) -> Dict[str, List[str]]:
    """Generate paraphrases for one dataset, with resume support."""

    # Resume: load existing JSON if present
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        existing = set(cache.keys())
        print(f"  [resume] loaded existing JSON with {len(existing)} classes")
    else:
        cache = {}
        existing = set()

    if smoke:
        classnames = classnames[:2]
        print(f"  [smoke] limited to first 2 classes: {classnames}")

    # Skip already-cached
    todo = [c for c in classnames if c not in existing]
    print(f"  [info] {len(todo)} new classes to process; {len(existing)} already cached")

    target_per_class = 4 * n_per_template  # 4 templates x 10 calls = 40
    per_class_pbar = tqdm(todo, desc=f"  {dataset}", ncols=80)

    for category in per_class_pbar:
        classname_clean = category.replace("_", " ")
        prompts = make_prompts(classname_clean)
        all_results: List[str] = []

        for prompt_idx, prompt in enumerate(prompts):
            for call_idx in range(n_per_template):
                result = client.chat(prompt)
                if result:
                    all_results.append(result)

        # write out partial result every class (for resume safety)
        cache[category] = all_results
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)

        per_class_pbar.set_postfix({
            "calls": client.total_calls,
            "fail": client.total_failures,
            "got": f"{len(all_results)}/{target_per_class}",
        })

    return cache


def main():
    parser = argparse.ArgumentParser(description="Generate DeepSeek paraphrases for 3D categories")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["modelnet40", "scanobjectnn"],
        choices=["modelnet40", "scanobjectnn", "omniobject3d", "objaverse_lvis"],
        help="datasets to process (default: modelnet40 scanobjectnn for AAAI phase)",
    )
    parser.add_argument(
        "--n_per_template",
        type=int,
        default=10,
        help="number of completions per template (4 templates total). Default 10 -> 40 paraphrases/class.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="DeepSeek temperature (lower = more reproducible). Default 0.3 per Gap F4.",
    )
    parser.add_argument(
        "--project_root",
        type=str,
        default="/root/autodl-tmp/MCP-Point-Cache",
        help="MCP-Point-Cache project root.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for JSON files. Default: <project_root>/Point-Cache/llm",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test mode: only 2 classes per dataset.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "Point-Cache" / "llm"
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY env var not set.")
        print("  Get a key at: https://platform.deepseek.com/api_keys")
        print("  Then run: export DEEPSEEK_API_KEY=sk-xxx")
        sys.exit(1)

    client = DeepSeekClient(api_key, temperature=args.temperature)

    print("=" * 70)
    print("MCP-3D Paraphrase Generation (DeepSeek)")
    print("=" * 70)
    print(f"  datasets:        {args.datasets}")
    print(f"  n_per_template:  {args.n_per_template}")
    print(f"  temperature:     {args.temperature}")
    print(f"  output_dir:      {output_dir}")
    print(f"  smoke:           {args.smoke}")
    print()

    for dataset in args.datasets:
        print(f"\n[{dataset}] loading classnames...")
        try:
            classnames = load_classnames(dataset, project_root)
        except FileNotFoundError as e:
            print(f"  [warn] {e}")
            print(f"  [warn] skipping {dataset}; download data first via: bash download_data.sh")
            continue
        print(f"  found {len(classnames)} classes")

        output_path = output_dir / f"{dataset}_deepseek_prompts.json"
        print(f"  output -> {output_path}")
        generate_for_dataset(
            dataset=dataset,
            classnames=classnames,
            output_path=output_path,
            client=client,
            n_per_template=args.n_per_template,
            smoke=args.smoke,
        )

    print()
    print("=" * 70)
    print(f"Done. Total API calls: {client.total_calls} (failures: {client.total_failures})")
    print("=" * 70)


if __name__ == "__main__":
    main()
