#!/usr/bin/env bash
# Fetch third-party code that has no license permitting redistribution,
# check out the commit used in the paper and apply our benchmarking patch.
#
#   bash third_party/fetch.sh            # all entries of third_party/sources.txt
#   bash third_party/fetch.sh FlashSparse
#
# Set <NAME>_URL (e.g. ACC_SPMM_REORDER_URL) to use a mirror.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCES="$ROOT/third_party/sources.txt"
PATCHES="$ROOT/third_party/patches"

want=("$@")
grep -v '^#' "$SOURCES" | while read -r name dest url commit _scope; do
  [[ -z "$name" ]] && continue
  if (( ${#want[@]} )) && [[ ! " ${want[*]} " =~ " $name " ]]; then continue; fi
  var="$(echo "$name" | tr 'a-z-' 'A-Z_')_URL"
  url="${!var:-$url}"
  target="$ROOT/$dest"
  if [[ -e "$target" ]]; then
    echo "[$name] $dest exists, skipping"
    continue
  fi
  if [[ "$commit" == "TBD" ]]; then
    echo "[$name] no pinned commit in sources.txt; run artifact_export/make_patches.sh first" >&2
    exit 1
  fi
  echo "[$name] cloning $url @ $commit"
  git init --quiet "$target"
  git -C "$target" remote add origin "$url"
  git -C "$target" fetch --quiet --depth 1 origin "$commit"
  git -C "$target" checkout --quiet FETCH_HEAD
  git -C "$target" submodule update --init --recursive --depth 1 --quiet
  if [[ -f "$PATCHES/$name.patch" ]]; then
    echo "[$name] applying patches/$name.patch"
    git -C "$target" apply "$PATCHES/$name.patch"
  fi
done
