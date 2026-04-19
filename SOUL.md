# SOUL.md — Research Norms & Code Standards

> Behavioral and technical guidelines for all agents working on DeepPhase-X.

## Research Philosophy

- **Rigor over speed**: Statistical significance before publication claims
- **Reproducibility**: Every experiment must be exactly reproducible from its config + seed + commit
- **Honesty**: Report negative results; "not significant" is a valid finding
- **Physics-first**: Every architectural decision should have a physical motivation or empirical justification

## Code Quality

- **Type hints** on all public function signatures
- **One-line docstrings** on public functions (WHAT it returns, not HOW)
- No dead code, no commented-out blocks, no placeholder TODOs in production
- Functions < 50 lines; refactor if longer
- All magic numbers in config files or named constants

## GPU Policy

- Training, inference, and data preprocessing **must** use GPU (`torch.cuda.is_available()` check)
- Simulation already supports CuPy via `src/simulation/backend.py` — use it
- Batch size should auto-scale to fill GPU memory (detect via `torch.cuda.mem_get_info()`)
- If GPU unavailable, log a warning but do not silently fall back to CPU without notification

## Branch & Merge Protocol

1. Create feature branch: `feature/<descriptive-name>`
2. Implement changes with tests
3. Run `pytest tests/` — all must pass
4. Multiple agents review the diff
5. Merge to master with descriptive commit message
6. No user approval needed for branch operations; user gates research plans and milestones only

## Reference Literature Protocol

When encountering useful literature during research:

1. **Download** PDF to `reference/参考文献/`
2. **First parse**: Extract into structured summary at `reference/参考文献_summaries/<paper_id>_summary.md`:
   - 摘要 (Abstract)
   - 核心方法 (Key Methods)
   - 主要结果 (Main Results)
   - 关键公式 (Key Formulas)
   - 与本项目的关系 (Relevance to this project)
3. **Register** in `reference/SUMMARY_INDEX.md` with one-line entry + file path
4. **Future access**: Read summary .md first; only re-parse original PDF when deep detail is needed

## Experiment Standards

- Every experiment gets a unique ID: `Exp{YYYYMMDD}_{type}_{NN}`
- Record: full config, random seed, git commit hash, GPU model, PyTorch version
- Checkpoint every N epochs; auto-resume from last checkpoint after interruption
- Delete checkpoints only after experiment successfully completes and results are saved

## Communication

- Weekly reports to user via WeChat (structured format: progress, findings, blockers, next steps)
- Research plan approval required before starting new research phases
- Milestone results require user review before proceeding to next phase
- All findings (positive or negative) documented in `experiments/logs/`
