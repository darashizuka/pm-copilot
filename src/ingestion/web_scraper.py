"""Scrape product teardowns and PRDs from blogs/Substack/GitHub. Upload to HuggingFace."""

import json
import os
import re
import time
from pathlib import Path

import feedparser
import requests
import trafilatura
from trafilatura.sitemaps import sitemap_search
from datasets import Dataset, concatenate_datasets, load_dataset

HF_DATASET_NAME = os.getenv("HF_DATASET_NAME", "shizukadara2/pm-copilot-raw")
HF_TOKEN = os.getenv("HF_TOKEN", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

MIN_WORD_COUNT = 500
REQUEST_DELAY = 1  # seconds between requests to avoid rate limits
TARGET_ARTICLE_LIMIT = 200
# Domains to crawl via sitemaps
DOMAINS_TO_CRAWL = [
    # "https://www.lennysnewsletter.com"
    # "https://www.svpg.com",
    # "https://review.firstround.com",
    # "https://caseyaccidental.com"
]

GITHUB_REPOS = [
    # ("PostHog", "posthog.com"),
    ("rust-lang", "rfcs"),
    ("reactjs", "rfcs"),
    ("backstage", "backstage"),
    ("aws", "aws-cdk"),
]

# Keywords that signal PM-relevant content
PM_KEYWORDS = [
    "product manag", "product strateg", "product sens", "product market fit",
    "prioritiz", "roadmap", "user research", "customer discovery",
    "retention", "churn", "onboarding", "activation",
    "a/b test", "experiment", "metric", "kpi", "okr", "north star",
    "prd", "product requirement", "feature spec",
    "user persona", "jobs to be done", "jtbd",
    "go-to-market", "gtm", "pricing", "freemium", "product-led growth", "plg",
    "stakeholder", "cross-functional", "sprint", "agile", "scrum",
    "conversion", "funnel", "engagement", "dau", "mau", "arpu",
    "marketplace", "two-sided", "network effect", "flywheel",
    "trade-off", "tradeoff", "prioritization framework", "rice score",
    "product decision", "product thinking", "product discovery",
    "mvp", "minimum viable", "iteration", "ship",
    "b2b", "b2c", "saas", "subscription",
]

# Keywords that indicate non-PM content
EXCLUDE_KEYWORDS = [
    "llm", "large language model", "transformer architecture", "fine-tun",
    "neural network", "deep learning", "machine learning model",
    "gpu cluster", "training pipeline", "inference endpoint",
    "prompt engineering", "rag pipeline", "vector database",
]


def is_pm_relevant(text: str, title: str = "") -> bool:
    combined = (title + " " + text).lower()
    pm_hits = sum(1 for kw in PM_KEYWORDS if kw in combined)
    exclude_hits = sum(1 for kw in EXCLUDE_KEYWORDS if kw in combined)

    # Need at least 3 PM keyword matches and more PM than exclude signals
    return pm_hits >= 3 and pm_hits > exclude_hits


def scrape_url(url: str) -> dict | None:
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None

        text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
        if not text or len(text.split()) < MIN_WORD_COUNT:
            return None

        metadata = trafilatura.extract_metadata(downloaded)
        title = metadata.title if metadata and metadata.title else url.split("/")[-1]

        if not is_pm_relevant(text, title):
            return None

        return {
            "id": url,
            "source": "web",
            "doc_type": "Strategy Note",
            "title": title,
            "url": url,
            "word_count": len(text.split()),
            "text": text,
        }
    except Exception as e:
        print(f"  Failed URL {url}: {e}")
        return None


def discover_sitemap_urls(domain: str) -> list[str]:
    print(f"  Discovering sitemap URLs for: {domain}")
    urls = sitemap_search(domain) or []
    article_urls = [u for u in urls if not re.search(r'/(about|archive|tags|podcast|subscribe)', u)]
    print(f"  Found {len(article_urls)} potential articles.")
    return article_urls


def discover_github_prds(owner: str, repo: str) -> list[str]:
    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

    for branch in ("main", "master"):
        url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            tree = resp.json().get("tree", [])

            prd_paths = [
                item["path"] for item in tree
                if item["type"] == "blob" and item["path"].endswith(".md")
                and re.search(r'(prd|rfc|spec|proposal|design[-_]doc|adr)', item["path"], re.IGNORECASE)
                and not re.search(r'(changelog|contributing|readme|license|code.of.conduct)', item["path"], re.IGNORECASE)
            ]
            print(f"  Found {len(prd_paths)} PRD/RFC files in {owner}/{repo} ({branch})")
            return prd_paths
        except Exception as e:
            print(f"  Failed GitHub API for {owner}/{repo} ({branch}): {e}")

    return []


def scrape_github_file(owner: str, repo: str, path: str) -> dict | None:
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{path}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 404:
            url = f"https://raw.githubusercontent.com/{owner}/{repo}/master/{path}"
            resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        text = resp.text
        if len(text.split()) < MIN_WORD_COUNT:
            return None
        return {
            "id": f"github:{owner}/{repo}/{path}",
            "source": "github",
            "doc_type": "RFC" if "rfc" in path.lower() else "PRD",
            "title": path.split("/")[-1],
            "url": f"https://github.com/{owner}/{repo}/blob/main/{path}",
            "word_count": len(text.split()),
            "text": text,
        }
    except Exception:
        return None


def get_existing_ids() -> set:
    try:
        ds = load_dataset(HF_DATASET_NAME, split="train")
        return set(ds["id"])
    except Exception:
        return set()


def scrape_all(push_to_hub: bool = True, dry_run: bool = False):
    existing_ids = get_existing_ids()
    print(f"Found {len(existing_ids)} existing docs on HuggingFace\n")
    results = []

    print("--- Crawling Web Domains via Sitemaps ---")
    for domain in DOMAINS_TO_CRAWL:
        urls = discover_sitemap_urls(domain)
        scraped, skipped, filtered = 0, 0, 0
        for url in urls:
            if len(results) >= TARGET_ARTICLE_LIMIT:
                print(f"\nReached target limit of {TARGET_ARTICLE_LIMIT} articles!")
                break
            if url in existing_ids:
                continue
            res = scrape_url(url)
            if res:
                results.append(res)
                scraped += 1
                print(f"  [{len(results)}] {res['title'][:60]} ({res['word_count']} words)")
            else:
                filtered += 1
            time.sleep(REQUEST_DELAY)
        print(f"  {domain}: {scraped} scraped, {filtered} filtered out\n")

    print("--- Crawling GitHub PRDs/RFCs ---")
    for owner, repo in GITHUB_REPOS:
        paths = discover_github_prds(owner, repo)
        for path in paths:
            gh_id = f"github:{owner}/{repo}/{path}"
            if gh_id not in existing_ids:
                res = scrape_github_file(owner, repo, path)
                if res:
                    results.append(res)
                    print(f"  [{len(results)}] {owner}/{repo}: {path.split('/')[-1]} ({res['word_count']} words)")

    print(f"\nTotal: {len(results)} new PM-relevant documents")

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
        print(f"Uploaded {len(results)} items to HuggingFace dataset: {HF_DATASET_NAME}")
    else:
        output_dir = Path.cwd() / "data" / "raw" / "web"
        output_dir.mkdir(parents=True, exist_ok=True)
        new_ds.save_to_disk(str(output_dir))
        print(f"Saved {len(results)} items locally to {output_dir}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Save locally instead of HuggingFace")
    parser.add_argument("--dry-run", action="store_true", help="Test without uploading")
    args = parser.parse_args()
    scrape_all(push_to_hub=not args.local, dry_run=args.dry_run)
