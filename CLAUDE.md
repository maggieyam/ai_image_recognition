# Rules for Claude

## Never commit data

This repo is public and open source. Datasets we use for research (OASIS, EMOTIC, ...) have their own licenses, and some (EMOTIC) only allow non-commercial research use, so they must never be redistributed through this repo.

- Never commit dataset files: images, annotations, ratings, CSVs, archives (.zip), or anything downloaded from a dataset.
- Never commit anything computed from a dataset, such as cached CLIP vectors or embeddings.
- Never commit trained weights unless the user explicitly approves that specific file.
- Keep all of it in `data/`, which is git-ignored. Scripts download or read datasets from there.
- Before every commit, check `git status` and the staged file list, and unstage anything that is data.

## Commits and pushes

- Never commit or push without the user's explicit permission. Approval to implement something is not approval to commit or push it.
- Implement, then stop so the user can review the change first.
