# Visualization Palette Consistency

When creating visualizations with matplotlib or seaborn for benchmark analysis, always use consistent, perceptually uniform color palettes.

## Default Palette Settings

Use these settings at the start of any visualization script:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set consistent style
sns.set_style("whitegrid")

# Default qualitative palette
sns.set_palette("muted")

# Standard figure size for reports
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
```

## Palette Selection Guide

**Qualitative (categorical data)**: `muted`
- Use for bar charts, line plots with distinct categories
- Example: comparing different configurations

**Circular (periodic data)**: `husl`
- Use when data wraps around (time of day, angles)
- Provides perceptually uniform spacing in circular color space

**Sequential/Heatmaps (continuous data)**: `magma`
- Use for heatmaps, sequential data visualization
- Perceptually uniform, colorblind-friendly
- Alternative: `viridis` for different aesthetic

## Example Usage

```python
# Categorical comparison
sns.set_palette("muted")
sns.barplot(data=df, x="config", y="throughput")

# Heatmap
sns.heatmap(data, cmap="magma")

# Circular data
sns.set_palette("husl")
plot_time_series(hourly_data)
```

## Rationale

- **Perceptual uniformity**: magma ensures equal visual distances represent equal data distances
- **Accessibility**: All chosen palettes are colorblind-friendly
- **Consistency**: Using same palettes across reports improves readability
- **Professionalism**: Cohesive color scheme throughout documentation
