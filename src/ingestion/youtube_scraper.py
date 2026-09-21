"""Scrape full-length podcast transcripts using yt-dlp for both discovery and subtitles.
Uploads directly to HuggingFace, no local storage needed."""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any

from datasets import Dataset, concatenate_datasets, load_dataset

HF_DATASET_NAME = os.getenv("HF_DATASET_NAME", "YOUR_HF_USERNAME/pm-copilot-raw")
HF_TOKEN = os.getenv("HF_TOKEN", "")

TARGET_CHANNELS = [
    "https://www.youtube.com/@LennysPodcast",
    "https://www.youtube.com/@ProductSchool",
    "https://www.youtube.com/@mindtheproduct",
    "https://www.youtube.com/@ycombinator",
    "https://www.youtube.com/@svpg",
]

PM_SEARCH_QUERIES = [
    "product management podcast full episode",
    "how to write a PRD product manager",
    "building AI products product lead",
    "product strategy case study",
]

EXCLUDE_KEYWORDS = ["#shorts", "clip", "teaser", "highlight", "trailer"]
MIN_WORD_COUNT = 1500
MIN_DURATION_SECS = 600


def ytdlp_cmd() -> list[str]:
    return [sys.executable, "-m", "yt_dlp"]


def run_ytdlp_discover(target_url: str, limit: int = 20) -> List[Dict[str, Any]]:
    cmd = ytdlp_cmd() + [
        "--dump-json",
        "--flat-playlist",
        "--playlist-end", str(limit),
        target_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    videos = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        try:
            data = json.loads(line)
            title = data.get("title", "")
            duration = data.get("duration") or 0

            if duration > 0 and duration < MIN_DURATION_SECS:
                continue
            if any(kw in title.lower() for kw in EXCLUDE_KEYWORDS):
                continue

            videos.append({
                "id": data.get("id"),
                "title": title,
                "duration": duration,
                "url": f"https://www.youtube.com/watch?v={data.get('id')}",
            })
        except json.JSONDecodeError:
            continue
    return videos


def get_transcript_ytdlp(video_id: str) -> str | None:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_template = os.path.join(tmpdir, "%(id)s")
        cmd = ytdlp_cmd() + [
            "--skip-download",
            "--write-auto-sub",
            "--write-sub",
            "--sub-lang", "en",
            "-o", output_template,
            f"https://www.youtube.com/watch?v={video_id}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        sub_files = list(Path(tmpdir).glob("*.vtt")) + list(Path(tmpdir).glob("*.srt"))

        if not sub_files:
            stderr = result.stderr.strip()
            if stderr:
                last_line = stderr.split("\n")[-1]
                print(f"    yt-dlp: {last_line[:120]}")
            return None

        sub_text = sub_files[0].read_text(encoding="utf-8", errors="ignore")
        return parse_vtt_to_text(sub_text)


def parse_vtt_to_text(vtt_content: str) -> str:
    lines = vtt_content.strip().split("\n")
    text_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line == "WEBVTT" or line.startswith("Kind:") or line.startswith("Language:"):
            continue
        if "-->" in line:
            continue
        if re.match(r"^\d+$", line):
            continue
        cleaned = re.sub(r"<[^>]+>", "", line)
        if cleaned and cleaned not in text_lines[-1:]:
            text_lines.append(cleaned)
    return " ".join(text_lines)


def discover_videos(limit_per_source: int = 20) -> List[Dict[str, Any]]:
    discovered = {}

    print("--- Scanning Target Channels ---")
    for channel_url in TARGET_CHANNELS:
        target = f"{channel_url}/videos"
        videos = run_ytdlp_discover(target, limit=limit_per_source)
        for v in videos:
            discovered[v["id"]] = v
        channel_name = channel_url.split("@")[-1]
        print(f"  [@{channel_name}]: {len(videos)} candidates (total unique: {len(discovered)})")

    print("\n--- Running Keyword Searches ---")
    for query in PM_SEARCH_QUERIES:
        target = f"ytsearch{limit_per_source}:{query}"
        videos = run_ytdlp_discover(target, limit=limit_per_source)
        for v in videos:
            discovered[v["id"]] = v
        print(f"  ['{query}']: {len(videos)} candidates (total unique: {len(discovered)})")

    return list(discovered.values())


def get_existing_ids() -> set:
    try:
        ds = load_dataset(HF_DATASET_NAME, split="train")
        return set(ds["id"])
    except Exception:
        return set()


def scrape_and_upload(
    limit_per_source: int = 20,
    push_to_hub: bool = True,
    dry_run: bool = False,
):
    existing_ids = get_existing_ids()
    print(f"Found {len(existing_ids)} existing transcripts on HuggingFace\n")

    candidates = discover_videos(limit_per_source=limit_per_source)
    new_candidates = [v for v in candidates if v["id"] not in existing_ids]
    print(f"\n{len(candidates)} total candidates, {len(new_candidates)} new\n")

    results = []
    for video in new_candidates:
        video_id = video["id"]
        try:
            full_text = get_transcript_ytdlp(video_id)
            if not full_text:
                print(f"  SKIP (no subs): {video['title'][:60]}")
                continue

            word_count = len(full_text.split())
            if word_count < MIN_WORD_COUNT:
                print(f"  SKIP ({word_count} words): {video['title'][:60]}")
                continue

            record = {
                "id": video_id,
                "source": "youtube",
                "doc_type": "Podcast Transcript",
                "title": video["title"],
                "url": video["url"],
                "word_count": word_count,
                "text": full_text,
            }
            results.append(record)
            print(f"  [{len(results)}] {video['title'][:60]} ({word_count} words)")

        except Exception as e:
            print(f"  FAIL ({type(e).__name__}): {video['title'][:60]}")

    print(f"\nRetrieved {len(results)} valid transcripts from {len(new_candidates)} candidates")

    if dry_run or not results:
        return results

    new_ds = Dataset.from_list(results)

    if push_to_hub:
        if existing_ids:
            old_ds = load_dataset(HF_DATASET_NAME, split="train")
            combined = concatenate_datasets([old_ds, new_ds])
            combined.push_to_hub(HF_DATASET_NAME, token=HF_TOKEN)
        else:
            new_ds.push_to_hub(HF_DATASET_NAME, token=HF_TOKEN)
        print(f"Pushed {len(results)} transcripts to {HF_DATASET_NAME}")

    print(f"Done. Total on HuggingFace: {len(existing_ids) + len(results)}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20, help="Videos per source")
    parser.add_argument("--dry-run", action="store_true", help="Test without uploading")
    parser.add_argument("--local", action="store_true", help="Skip HuggingFace upload")
    args = parser.parse_args()
    scrape_and_upload(
        limit_per_source=args.limit,
        push_to_hub=not args.local,
        dry_run=args.dry_run,
    )
