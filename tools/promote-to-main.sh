#!/usr/bin/env bash
#
# promote-to-main.sh — promote core JUG changes from dev to main.
#
# main is curated core-machinery-only: notebooks and the speed baseline are
# never carried over. main's history was filter-branch rewritten, so we do NOT
# `git merge dev` (that 3-way-merges against rewritten history). Instead we
# check out only the wanted paths from dev into a throwaway worktree on main,
# commit, and push. The dev working tree is never switched (notebook
# skip-worktree edits stay put).
#
# Usage:
#   tools/promote-to-main.sh [-n] [-m "commit message"] [path ...]
#
#   -n            dry run: show what would be promoted, change nothing
#   -m MSG        commit message (otherwise an editor opens / a default is used)
#   path ...      restrict to these paths (default: every dev/main diff except
#                 the excluded globs below)
#
# Examples:
#   tools/promote-to-main.sh -n                       # preview the promotion
#   tools/promote-to-main.sh -m "fix: cache key len"  # promote all core diffs
#   tools/promote-to-main.sh jug/utils/geom_cache.py  # promote one file
#
set -euo pipefail

DEV_BRANCH="dev"
MAIN_BRANCH="main"
REMOTE="origin"

# Paths that must NEVER land on main (extended-glob patterns).
EXCLUDE_GLOBS=( 'notebooks/*' '.jug_speed_baseline.json' 'tools/promote-to-main.sh' )

dry_run=0
msg=""
declare -a want_paths=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n) dry_run=1; shift ;;
    -m) msg="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    --) shift; while [[ $# -gt 0 ]]; do want_paths+=("$1"); shift; done ;;
    *)  want_paths+=("$1"); shift ;;
  esac
done

# Run from repo root regardless of CWD.
repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

is_excluded() {
  local p="$1" g
  shopt -s extglob
  for g in "${EXCLUDE_GLOBS[@]}"; do
    # shellcheck disable=SC2053
    [[ "$p" == $g ]] && return 0
  done
  return 1
}

echo "Fetching $REMOTE ..."
git fetch --quiet "$REMOTE"

# Fast-forward local main to origin/main if behind (origin/main is often ahead).
behind=$(git rev-list --count "$MAIN_BRANCH..$REMOTE/$MAIN_BRANCH" 2>/dev/null || echo 0)
ahead=$(git rev-list --count "$REMOTE/$MAIN_BRANCH..$MAIN_BRANCH" 2>/dev/null || echo 0)
if [[ "$behind" -gt 0 && "$ahead" -gt 0 ]]; then
  echo "ERROR: local $MAIN_BRANCH and $REMOTE/$MAIN_BRANCH have diverged "\
       "($ahead local / $behind remote). Resolve manually." >&2
  exit 1
fi
if [[ "$behind" -gt 0 ]]; then
  echo "Fast-forwarding local $MAIN_BRANCH to $REMOTE/$MAIN_BRANCH (+$behind) ..."
  git update-ref "refs/heads/$MAIN_BRANCH" "$REMOTE/$MAIN_BRANCH"
fi

# Determine candidate paths: explicit args, or the full dev/main diff.
declare -a candidates=()
if [[ ${#want_paths[@]} -gt 0 ]]; then
  candidates=("${want_paths[@]}")
else
  mapfile -t candidates < <(git diff --name-only "$MAIN_BRANCH" "$DEV_BRANCH")
fi

# Filter out excluded paths.
declare -a promote=()
declare -a skipped=()
for p in "${candidates[@]}"; do
  [[ -z "$p" ]] && continue
  if is_excluded "$p"; then skipped+=("$p"); else promote+=("$p"); fi
done

if [[ ${#promote[@]} -eq 0 ]]; then
  echo "Nothing to promote (after exclusions). Up to date."
  [[ ${#skipped[@]} -gt 0 ]] && printf '  excluded: %s\n' "${skipped[@]}"
  exit 0
fi

echo
echo "Will promote $DEV_BRANCH -> $MAIN_BRANCH:"
printf '  + %s\n' "${promote[@]}"
[[ ${#skipped[@]} -gt 0 ]] && printf '  - (excluded) %s\n' "${skipped[@]}"
echo

if [[ "$dry_run" -eq 1 ]]; then
  echo "Dry run: no worktree created, nothing committed."
  exit 0
fi

# Throwaway worktree on main so the primary dev tree is never switched.
wt=$(mktemp -d "${TMPDIR:-/tmp}/jug-promote.XXXXXX")
cleanup() { git worktree remove --force "$wt" 2>/dev/null || true; rmdir "$wt" 2>/dev/null || true; }
trap cleanup EXIT

git worktree add --quiet "$wt" "$MAIN_BRANCH"
git -C "$wt" checkout "$DEV_BRANCH" -- "${promote[@]}"

if git -C "$wt" diff --cached --quiet; then
  echo "No effective change on $MAIN_BRANCH (already identical). Nothing committed."
  exit 0
fi

if [[ -z "$msg" ]]; then
  msg="promote core changes from $DEV_BRANCH"$'\n\n'"$(printf '%s\n' "${promote[@]}")"
fi

git -C "$wt" commit --quiet -m "$msg"
echo "Committed on $MAIN_BRANCH:"
git -C "$wt" --no-pager log -1 --oneline

echo "Pushing to $REMOTE/$MAIN_BRANCH ..."
git -C "$wt" push --quiet "$REMOTE" "$MAIN_BRANCH"
echo "Done."
