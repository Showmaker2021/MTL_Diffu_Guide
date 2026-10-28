#!/usr/bin/env python3
"""Download selected CVRPLIB Set-X instances from the official CVRPLIB website."""

from __future__ import annotations

import pathlib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

INSTANCES = [
    "X-n524-k153",
    "X-n536-k96",
    "X-n548-k50",
    "X-n561-k42",
    "X-n573-k30",
    "X-n586-k159",
    "X-n599-k92",
    "X-n613-k62",
    "X-n627-k43",
    "X-n641-k35",
    "X-n655-k131",
    "X-n670-k130",
    "X-n685-k75",
    "X-n701-k44",
    "X-n716-k35",
    "X-n733-k159",
    "X-n749-k98",
    "X-n766-k71",
    "X-n783-k48",
    "X-n801-k40",
    "X-n819-k171",
    "X-n837-k142",
    "X-n856-k95",
    "X-n876-k59",
    "X-n895-k37",
    "X-n916-k207",
    "X-n936-k151",
    "X-n957-k87",
    "X-n979-k58",
    "X-n1001-k43",
]

BASE_URL = "https://galgos.inf.puc-rio.br"
PAGES = [
    f"{BASE_URL}/cvrplib/en/instances",
    f"{BASE_URL}/cvrplib/en/instances/1",
]
OUT_DIR = pathlib.Path(__file__).resolve().parent / "data" / "CVRP-LIB"
USER_AGENT = "Mozilla/5.0 (compatible; CVRPLIBDownloader/1.0)"


class LinkParser(HTMLParser):
    """Collect links and their surrounding table-row text."""

    def __init__(self):
        super().__init__()
        self.links = []
        self.current_href = None
        self.current_text = []
        self.row_text = []
        self.row_links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "tr":
            self.row_text = []
            self.row_links = []
        elif tag == "a":
            self.current_href = attrs.get("href")
            self.current_text = []

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
        self.row_text.append(text)
        if self.current_href is not None:
            self.current_text.append(text)

    def handle_endtag(self, tag):
        if tag == "a" and self.current_href is not None:
            self.row_links.append(
                (self.current_href, " ".join(self.current_text))
            )
            self.current_href = None
            self.current_text = []

        elif tag == "tr":
            row = " ".join(self.row_text)
            for href, text in self.row_links:
                self.links.append((row, href, text))


def fetch(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def find_download_links():
    """Index instance rows and their download links from official listing pages."""
    index = {}

    for page_url in PAGES:
        try:
            html = fetch(page_url).decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"[warning] Cannot read listing {page_url}: {exc}")
            continue

        parser = LinkParser()
        parser.feed(html)

        for row, href, text in parser.links:
            for name in INSTANCES:
                if re.search(rf"(?<![A-Za-z0-9_-]){re.escape(name)}(?![A-Za-z0-9_-])", row):
                    absolute_url = urllib.parse.urljoin(page_url, href)
                    index.setdefault(name, []).append(
                        (absolute_url, text, row)
                    )

    return index


def download(name: str, candidates, out_dir: pathlib.Path):
    destination = out_dir / f"{name}.vrp"

    if destination.exists() and destination.stat().st_size > 0:
        print(f"[skip] {name}: already exists")
        return True

    for url, link_text, row in candidates:
        # Prefer links that look like file-download actions.
        if not any(word in (url + " " + link_text).lower()
                   for word in ("download", "file", ".vrp", "instance")):
            continue

        temp_path = destination.with_suffix(".vrp.part")
        try:
            data = fetch(url)

            # Reject HTML error pages saved under a .vrp filename.
            prefix = data[:500].lstrip().lower()
            if not data or b"<html" in prefix or b"<!doctype html" in prefix:
                print(f"[warning] {name}: link returned HTML, not a VRP file")
                continue

            temp_path.write_bytes(data)
            temp_path.replace(destination)
            print(f"[ ok ] {name} ({len(data):,} bytes)")
            return True

        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            temp_path.unlink(missing_ok=True)
            print(f"[warning] {name}: failed URL {url}: {exc}")

    print(f"[error] {name}: no working download link found")
    return False


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR}")
    print("Reading official CVRPLIB instance listings...")

    index = find_download_links()
    succeeded = []
    failed = []

    for i, name in enumerate(INSTANCES, 1):
        print(f"\n[{i}/{len(INSTANCES)}] {name}")

        candidates = index.get(name, [])
        if not candidates:
            print("[error] Instance or download link not found in listing")
            failed.append(name)
            continue

        if download(name, candidates, OUT_DIR):
            succeeded.append(name)
        else:
            failed.append(name)

        time.sleep(0.3)

    print(f"\nFinished: {len(succeeded)} succeeded, {len(failed)} failed.")

    if failed:
        print("Failed instances:")
        for name in failed:
            print(f"  {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
