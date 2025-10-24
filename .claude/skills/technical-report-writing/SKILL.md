# Technical Report Writing Standards

When creating technical reports or benchmark analysis documents, follow these guidelines to maintain a neutral, fact-based tone appropriate for technical documentation.

## Tone and Language

### Avoid Marketing Language
**DON'T use:**
- Superlatives: "comprehensive", "dramatic", "substantial", "superior", "excellent"
- Emphatic prefixes: "Key Findings", "Important Results", "Critical Insights"
- Recommendations: "strongly recommend", "should use", "best practice"
- Value judgments: "acceptable", "unacceptable", "good enough"
- Enthusiastic framing: "revealing", "exciting", "impressive"

**DO use:**
- Neutral descriptors: "shows", "indicates", "demonstrates", "observes"
- Specific measurements: "+32.7% increase" instead of "dramatic increase"
- Factual comparisons: "higher throughput" instead of "superior performance"
- Observations: "Observations" instead of "Key Findings"

### Avoid Inappropriate Emphasis
- **DON'T** capitalize words for emphasis (e.g., "HIGHER throughput")
- **DON'T** use bold for emphasis in body text excessively
- **DO** use standard formatting and let the data speak for itself

### Avoid Overgeneralization
- **DON'T** generalize from small sample sizes
  - ❌ "Smaller models show behavior X" (when you tested 1 small model)
  - ✅ "Qwen3-0.6B shows behavior X"
- **DON'T** use categorical statements without data
  - ❌ "This approach is better for production"
  - ✅ "This approach shows 32% higher throughput in this test"

## Structure

### Observations vs Findings
- Use "Observations" instead of "Key Findings" or "Critical Results"
- Present data neutrally without editorializing

### Comparisons
- Always specify what you're comparing (A vs B)
- Use specific model/configuration names, not categories
- Include the measurement or percentage difference

### Conclusions
- Base conclusions strictly on the data presented
- Avoid recommendations unless explicitly requested
- Focus on "what" happened, not "what should" happen

## Examples

### Bad (Marketing-focused):
```
KEY FINDINGS:

The CPU offload approach delivers DRAMATIC performance improvements,
with throughput SIGNIFICANTLY higher than baseline. We strongly recommend
this approach for production deployments of larger models.
```

### Good (Technical/neutral):
```
Observations:

CPU offload shows 32.7% higher throughput for Qwen3-8B compared to baseline
(69.9 vs 52.7 tok/s). Qwen3-0.6B shows minimal difference (-1.9%).
```

## Checklist Before Finalizing Report

- [ ] No marketing superlatives (comprehensive, dramatic, etc.)
- [ ] No capitalization for emphasis
- [ ] "Observations" instead of "Key Findings"
- [ ] Specific model names, not generalizations
- [ ] Measurements included with all comparisons
- [ ] No recommendations unless requested
- [ ] Neutral, fact-based tone throughout
