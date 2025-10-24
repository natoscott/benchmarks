# Lessons Learned: vLLM KV Cache CPU Offload Analysis

This document captures issues encountered during the October 2025 analysis and improvements made for future work.

## Issues Encountered

### 1. Marketing-Focused Language (Multiple Corrections)
**Problem**: Report used enthusiastic, marketing-style language inappropriate for technical documentation.
- Used words like "comprehensive", "dramatic", "substantial", "superior"
- Capitalized words for emphasis (e.g., "HIGHER throughput")
- Added "Key Findings" headers and recommendations
- Required 3+ rounds of corrections to achieve neutral tone

**Root Cause**: No established guidelines for technical report tone.

**Solution**: Created `skills/technical-report-writing/SKILL.md` with:
- List of words to avoid
- Examples of neutral alternatives
- Checklist for report review
- Guidelines on avoiding overgeneralization

### 2. Visualization Issues

#### a) Didn't Follow Existing Standards
**Problem**: Used custom hex colors instead of checking `skills/visualization-palette/SKILL.md`.
- Should have used `sns.set_palette("muted")` from the start
- Required regenerating all visualizations

**Solution**:
- Added reminder to check skills files FIRST
- Created data validation checklist

#### b) Missing Data Points in Graphs
**Problem**: All graphs showed no data points for model-specific plots.
- Used column `concurrency` which was all nulls
- Should have used `concurrency.1`
- Affected 3 separate visualization files

**Root Cause**: Didn't validate DataFrame before plotting.

**Solution**: Created `skills/data-validation-checklist/SKILL.md` with:
- Pre-visualization DataFrame inspection checklist
- Post-visualization visual inspection requirements
- Common pitfalls and how to avoid them

#### c) Text Overlays Obscuring Details
**Problem**: Added summary text boxes on graphs that obscured data.

**Solution**:
- Remove text overlays from data visualizations
- Use separate summary figures if needed

#### d) Graph Readability Issues
**Problem**: Performance improvements graph had:
- Too many x-axis labels (9 concurrency levels)
- Extreme y-axis values making bars illegible (-205%)

**Solution**:
- Reduce tick density (show every other label)
- Set reasonable y-axis limits

### 3. Inappropriate Generalizations
**Problem**: Made broad statements from sample size of 2.
- "Smaller models do X" when only tested Qwen3-0.6B
- "Larger models do Y" when only tested Qwen3-8B

**Solution**:
- Added guideline to technical-report-writing skill
- Always use specific model names
- Avoid categorical statements without sufficient data

### 4. Repository Cleanliness
**Problem**: Accumulated 21 untracked intermediate files by end of session.
- No .gitignore created upfront
- 149MB `guidellm_data.csv` file left in working directory
- Unclear which files were essential vs intermediate

**Solution**: Created `skills/repo-cleanliness/SKILL.md` with:
- Create .gitignore at project start
- Progressive cleanup strategy (clean as you go)
- File tiering system (essential/intermediate/temporary)
- Standard .gitignore patterns for data analysis

## Improvement Plan

### For Next Analysis Session

1. **Start with Skills Review**
   - Read `skills/technical-report-writing/SKILL.md` before writing
   - Read `skills/visualization-palette/SKILL.md` before plotting
   - Read `skills/repo-cleanliness/SKILL.md` at project start

2. **Create .gitignore Early**
   - First thing after starting analysis
   - Add common patterns immediately

3. **Validate Data Before Plotting**
   - Always inspect DataFrame structure
   - Check for null values in plot columns
   - Verify filtered data exists

4. **Inspect Visualizations After Generation**
   - Open and view every PNG file
   - Check that data points are visible
   - Verify colors match palette standards

5. **Use Neutral Technical Language**
   - Avoid superlatives and marketing terms
   - Use "Observations" not "Key Findings"
   - Don't capitalize for emphasis
   - Be specific, avoid overgeneralization

6. **Progressive Cleanup**
   - Review `git status` after each major step
   - Commit essential files immediately
   - Add intermediate files to .gitignore
   - Don't accumulate >10 untracked files

## Success Metrics for Next Time

- [ ] No marketing language corrections needed
- [ ] All visualizations show data on first generation
- [ ] .gitignore created before first intermediate file
- [ ] <5 untracked files at project end
- [ ] No overgeneralizations from small samples
- [ ] Visualization palette standards followed from start

## New Skills Created

1. **technical-report-writing**: Guidelines for neutral, fact-based technical writing
2. **data-validation-checklist**: Pre/post checks for data analysis and visualization
3. **repo-cleanliness**: Progressive cleanup and .gitignore management

These skills should be consulted at the START of future analysis work, not discovered mid-project.

---

*Created: October 24, 2025*
*Project: vLLM KV Cache CPU Offload Evaluation*
