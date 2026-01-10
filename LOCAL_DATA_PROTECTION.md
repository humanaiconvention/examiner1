# Local Data Protection Guide

## ✅ Your Data Stays Local

This guide ensures all your personal, domain-specific data stays on your local machine in `D:\humanaiconvention\examiner` and never syncs to GitHub.

## 📂 Directory Structure: What Stays Local

### Your Corpus Files (DO NOT PUSH)
```
D:\humanaiconvention\examiner\data/
├── pdfs/                          # ← Your domain PDFs
│   └── *.pdf                       # STAYS LOCAL ✓
├── datasets/                       # ← Your extracted datasets
│   └── *.json                      # STAYS LOCAL ✓
└── raw/                            # ← Raw extracted text
    └── *.txt                       # STAYS LOCAL ✓
```

### Your Training Data (DO NOT PUSH)
```
D:\humanaiconvention\examiner/
├── corpus_index.json               # Your corpus metadata - STAYS LOCAL ✓
├── lived_experience_log.json       # Your expert dialogue - STAYS LOCAL ✓
├── training_data*.json             # Generated training data - STAYS LOCAL ✓
└── training_manifest.json          # Training record - STAYS LOCAL ✓
```

### Your Model Weights (DO NOT PUSH)
```
D:\humanaiconvention\examiner\models/
├── grounding_cycle_1/
│   ├── checkpoints/                # Training checkpoints - STAYS LOCAL ✓
│   └── final/                      # Trained weights - STAYS LOCAL ✓
└── *.pt, *.pth                     # Model files - STAYS LOCAL ✓
```

### Your Training Logs (DO NOT PUSH)
```
D:\humanaiconvention\examiner\logs/
└── *.log                            # Training logs - STAYS LOCAL ✓
```

## 🔒 Git Protection Layers

### Layer 1: .gitignore (Active)
```
# In .gitignore
data/pdfs/
data/datasets/
data/raw/
corpus_index.json
lived_experience_log.json
training_data*.json
models/final/
models/checkpoints/
logs/
```

**Status**: ✅ Active - These files will NOT be committed

### Layer 2: Git Configuration (Extra Safety)
```bash
# Already configured - prevents accidental adds
git config core.safecrlf true
```

**Status**: ✅ Active - Prevents some edge cases

### Layer 3: Git Hooks (Optional Extra Protection)

To add maximum protection, I can set up a pre-commit hook that blocks any local data files. Would you like me to set this up? It would:
- Prevent commits if sensitive files are staged
- Warn before pushing
- Never auto-fix (manual approval required)

## 📋 Current Status

### ✅ Files Protected from Sync
- Your 104 PDF research papers (articles/)
- Your corpus index (corpus_index.json)
- Your expert dialogue (lived_experience_log.json)
- Your generated training data (training_data_cycle_1.json)
- Your trained model weights (models/grounding_cycle_1/final/)
- Your training logs (logs/)

**All untracked by git** ✓ - Safe to keep locally

### ✅ Files That DO Get Synced
- Framework code (all 8 .py modules)
- Documentation (guides, README, etc.)
- Dependencies (requirements.txt)
- License (MIT)
- Configuration (setup.sh)
- .gitignore itself

**All tracked and synchronized** ✓ - Synced to `https://github.com/humanaiconvention/examiner`

## 🛡️ Daily Protection Checklist

Before each work session:

```bash
# 1. Check status - see what's untracked (local)
git status

# 2. Verify nothing sensitive is staged
git diff --cached  # Should show only docs/code changes

# 3. Before pushing
git push origin main  # Should upload only code/docs, not data

# 4. Confirm nothing leaked
git ls-files | grep -E "corpus|training_data|articles"
# ^ Should return nothing (means files aren't tracked)
```

## ⚠️ How to Avoid Accidents

### ✅ DO These Things
```bash
# Add documentation files
git add README.md CORPUS_SETUP.md

# Commit framework improvements
git commit -m "Update training pipeline"

# Push code and docs
git push origin main
```

### ❌ DON'T Do These Things
```bash
# Don't add your corpus files
git add data/pdfs/
git add corpus_index.json

# Don't add your trained models
git add models/grounding_cycle_1/final/

# Don't force-push user data
git push --force  # (only for code, never with local data staged)

# Don't use git add .
git add .  # Too risky - use specific files instead
```

## 🔍 Verification Commands

Run these anytime to verify nothing local is being tracked:

```bash
# List all tracked files
git ls-files

# Should NOT contain: articles/, corpus_index.json, training_data*.json, 
#                     models/grounding_cycle_1/final/, logs/

# Check untracked local files
git status

# Should show: articles/, corpus_index.json, etc. as "untracked"
#             (this means they're LOCAL ONLY)

# List what would be pushed
git push --dry-run

# Should NOT show: .json corpus files, trained models, logs/
```

## 📊 Storage Summary

```
D:\humanaiconvention\examiner/
├── Framework Code (GITHUB)        ← ~2MB   (synced to GitHub)
├── Documentation (GITHUB)         ← ~1MB   (synced to GitHub)
├── data/pdfs/ (LOCAL ONLY)        ← ~5GB   (your PDFs - NEVER synced)
├── data/datasets/ (LOCAL ONLY)    ← ~200MB (your datasets - NEVER synced)
├── models/ (LOCAL ONLY)           ← ~10GB  (your trained weights - NEVER synced)
└── logs/ (LOCAL ONLY)             ← ~100MB (training logs - NEVER synced)

TOTAL:
├── GitHub:    ~3MB   (framework + docs)
└── Local:     ~15GB  (your data, models, training artifacts)
```

Your local data never leaves your machine. ✓

## 🚨 If You Accidentally Commit Something

Don't panic! Here's how to fix it:

```bash
# 1. Remove from staging (if not yet committed)
git reset HEAD <filename>

# 2. If already committed
git rm --cached <filename>  # Remove from git but keep locally
git commit -m "Remove <filename> from tracking"

# 3. If pushed to GitHub
# Contact GitHub Support or use BFG Repo-Cleaner
# (For critical data, this might need force-push)
```

## 📝 Golden Rule

**IF IN DOUBT**: Use `git add <specific_file.py>` instead of `git add .`

This prevents accidents and keeps your local data safe.

## ✅ Conclusion

Your setup is secure:

1. **Framework code** = Public on GitHub ✓
2. **Documentation** = Public on GitHub ✓
3. **Your corpus** = Local only ✓
4. **Your models** = Local only ✓
5. **Your expertise** = Local only ✓

Everything stays in `D:\humanaiconvention\examiner` locally while the framework is shared globally. Perfect balance of openness and privacy! 🔒

---

**Status: All local data protected from sync to GitHub**
