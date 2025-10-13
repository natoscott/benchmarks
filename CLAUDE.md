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
