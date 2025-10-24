# Repository Cleanliness and Progressive Cleanup

Keep repositories clean by managing intermediate files proactively throughout analysis work, not just at the end.

## Start Clean

### Create .gitignore Early
**When**: At the beginning of any analysis project
**What**: Add common intermediate file patterns

```bash
# Create .gitignore before generating intermediate files
cat > .gitignore << 'EOF'
# Intermediate analysis files
*_intermediate.csv
*_temp.csv
*_test.py
temp_*.py

# Large data exports
*_timeseries.csv
*_raw.csv
guidellm_data.csv

# Internal notes
FINDINGS*.md
NOTES*.md
TODO*.md
EOF
```

## Progressive Cleanup Strategy

### Tier 1: Essential (COMMIT these)
- Final report (REPORT.md, README.md)
- Final analysis scripts (produce the results in the report)
- Final visualization scripts (produce the graphs in the report)
- Key result CSVs (small, < 50KB, used in final analysis)
- Visualizations (PNG files referenced in report)
- Source data (PCP archives, raw measurements if compressed)

### Tier 2: Intermediate (GITIGNORE these)
- Scripts used during development/exploration
- Intermediate CSV files from analysis pipeline
- Test scripts and experiments
- Internal documentation/notes

### Tier 3: Temporary/Large (GITIGNORE these)
- Raw data exports > 1MB
- Cache files
- Extracted timeseries data
- Duplicate/old versions of files

## Identify Files to Ignore

### During Analysis
After each major analysis step:
```bash
# Check what's untracked
git status

# For each untracked file, ask:
# - Is this needed to reproduce the final results? → COMMIT
# - Is this intermediate/temporary? → ADD TO .gitignore
# - Is this large (>1MB) and regeneratable? → ADD TO .gitignore
```

### File Naming Convention
Help identify file purpose by naming:
- `final_*.csv` - Final results to commit
- `intermediate_*.csv` - Pipeline step, add to .gitignore
- `temp_*.csv` - Temporary, add to .gitignore
- `analyze_*.py` - If used in final report, commit; else ignore
- `test_*.py` - Always ignore

## Clean As You Go

### After Each Analysis Phase
1. Review `git status`
2. Commit essential files
3. Update .gitignore for intermediate files
4. Don't accumulate >20 untracked files

### Example Workflow
```bash
# After extracting raw data
git add analyze_extract.py final_results.csv
git add .gitignore  # Added: raw_export.csv
git commit -m "Extract benchmark results"

# After processing
git add analyze_process.py processed_results.csv
git add .gitignore  # Added: intermediate_*.csv
git commit -m "Process and clean results"

# After visualization
git add visualize.py results_chart.png
git commit -m "Create visualizations"
```

## .gitignore Patterns for Data Analysis

Standard patterns to include:
```gitignore
# Analysis intermediate files
intermediate_*.csv
temp_*.csv
test_*.csv
*_temp.py
*_test.py

# Large data exports
*_timeseries.csv
*_raw_*.csv
*_export.csv
guidellm_data.csv

# Cache and temporary
__pycache__/
*.pyc
.ipynb_checkpoints/

# Notes and planning
*_NOTES.md
*_TODO.md
FINDINGS*.md
```

## End-of-Project Review

Before finalizing:
```bash
# List all tracked files
git ls-files

# Review for:
# - Old versions that should be removed
# - Test files accidentally committed
# - Large files that should be in LFS or removed

# Check repository size
du -sh .git

# If >100MB, investigate:
git ls-files | xargs ls -lh | sort -k5 -h | tail -20
```

## Anti-Patterns to Avoid

❌ **Don't**: Leave 20+ untracked files at end of project
✅ **Do**: Add to .gitignore progressively

❌ **Don't**: Commit intermediate_*.csv files
✅ **Do**: Only commit final_*.csv files

❌ **Don't**: Commit 149MB CSV files
✅ **Do**: Add to .gitignore, use compressed formats

❌ **Don't**: Create .gitignore only when asked to clean up
✅ **Do**: Create .gitignore at project start

## Quick Cleanup Command

When you have many intermediate files:
```bash
# Review untracked files by size
git status --short | grep '^??' | awk '{print $2}' | xargs ls -lh 2>/dev/null | sort -k5 -h

# Add common patterns to .gitignore in one go
cat >> .gitignore << 'EOF'
# Generated during analysis cleanup
annotated_*.csv
connector_*.csv
extracted_*.csv
identified_*.csv
*_analysis.csv
EOF
```
