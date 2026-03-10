# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with benchmark results and system performance data in is repository.

## Input

This git repository contains results from benchmarking features of the
vLLM and llm-d projects.

Performance Co-Pilot (PCP) - a toolkit for system-level performance
monitoring and analysis - provides a unifying abstraction for all
performance data in the system under test.  It provides many tools for
interrogating, retrieving and processing the recorded data.

Important tools relative to these benchmarks are pcp2arrow(1) for creating
Parquet format from PCP archives, guidellm2pcp(1) and vllmbench2pcp(1)
for representing benchmark results from guidellm and 'vllm bench' as
PCP metrics within PCP archives.

The recorded PCP archives contain metrics from the kernel and hardware,
especially nvidia or AMD GPU metrics.  All available Prometheus metrics
from vLLM and llm-d are sampled using the openmetrics PMDA and recorded
in the archives.

Each top level directory in the git repo represents an independant
benchmark scenario.

## Output

The Performance Co-Pilot archives in each directory capture the state
of all systems involved in benchmarking, as well as the results of the
benchmarks.  Using this input data the results are interpreted, compared,
collated and distilled down into a single markdown file and associated
graphs that explain the results.

A common approach to the data analysis creation of visuals is to convert
the recorded time series metrics into Parquet form for ease of loading
into Python Pandas dataframes.  Other python libraries can then be used
for analysis (such as scikit) and graphing (matplotlib, seaborn, plotly
and bokeh, for example).

The goal is to produce concise, insightful and easily understood reports
explaining the results of the benchmarks.  At the end of each session in
which a report is created, always consider what might have been done to
improve the process for the human analyst - suggest new tools, additional
metrics, and alternative benchmarking strategies to support the goal.

## Project Standards

This repository has established standards documented in `.claude/skills/`:

- **visualization-palette**: All visualizations must use seaborn's "muted" palette for categorical data, "magma" for heatmaps, and standard figure sizes (14x8, font size 11). Consult `.claude/skills/visualization-palette/SKILL.md` when creating any matplotlib/seaborn visualizations.

- **technical-report-writing**: Guidelines for professional, measured technical writing. Consult `.claude/skills/technical-report-writing/SKILL.md` when writing or updating reports.

- **pcp-benchmark-analysis**: Best practices for analyzing PCP archives and correlating system metrics with benchmark results. Consult `.claude/skills/pcp-benchmark-analysis/SKILL.md` when working with Performance Co-Pilot data.

- **data-validation-checklist**: Validation steps to ensure data quality and consistency. Consult `.claude/skills/data-validation-checklist/SKILL.md` before finalizing analysis.

- **repo-cleanliness**: Standards for maintaining clean repository structure. Consult `.claude/skills/repo-cleanliness/SKILL.md` when creating new files or organizing results.

Always review the relevant skill documentation before starting work in these areas.

## Writing Style Guidelines

Reports in this repository must maintain a **neutral, factual tone** that
presents data without emotional characterization or subjective judgment.

### Avoid Opinionated Language

Do NOT use adjectives that imply subjective judgment or emotional response:

- **Avoid**: concerning, catastrophic, severe, unexpected, alarming,
  troubling, worrying, problematic, dramatic, disappointing
- **Instead**: Use quantified descriptors or neutral factual statements

### Use Quantified Descriptors

Always prefer specific numbers and measurements over qualitative adjectives:

**Good examples:**
- "14B model shifts from +0.6% to -8.1% (-8.7 pp regression)"
- "32B-AWQ shows -56.2% throughput loss"
- "Overhead reduced from -29.1% to -3.0% (+26.1 pp improvement)"
- "9-14 individual CPUs averaged >80% saturation"

**Avoid:**
- "severely degraded performance"
- "concerning regression"
- "catastrophic throughput loss"

### When "Significant" and "Substantial" are Acceptable

These words can be used when describing **quantified changes** where the
magnitude is objectively large:

**Acceptable:**
- "vLLM 0.14.1 significantly improves native CPU offload for small models
  (0.6B: +26.1 pp improvement)"
- "substantial changes to the KV offloading implementation"

**Not acceptable:**
- "significant problems with the implementation"
- "substantially worse than expected"

The key difference: use "significant/substantial" to describe the **size**
of a measured change, not to characterize something as good or bad.

### Examples from Report Updates

**Before:**
- "Concerning regressions for larger models"
- "Catastrophic degradation for 32B-AWQ (-56.2%)"
- "Severe load hotspotting across CPUs"
- "Most severe overhead observed"

**After:**
- "Regressions for larger models"
- "32B-AWQ shows -56.2% degradation"
- "High load hotspotting across CPUs"
- "Largest overhead observed"

### General Principles

1. **Present facts**: State what the data shows
2. **Quantify changes**: Use percentages, absolute numbers, ranges
3. **Be precise**: Avoid vague characterizations
4. **Stay neutral**: Let the reader draw conclusions from the data

The reports serve technical audiences who value objective analysis over
editorial commentary. When in doubt, remove the adjective and let the
numbers speak.
