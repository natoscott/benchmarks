# Data Validation and Visualization Checklist

When performing data analysis and creating visualizations, always validate your work to catch common issues early.

## Before Creating Visualizations

### 1. Inspect DataFrame Structure
```python
# Always check:
print(df.columns.tolist())  # Column names
print(df.dtypes)            # Data types
print(df.head(10))          # First rows
print(df.shape)             # Dimensions
print(df.isnull().sum())    # Missing values
```

### 2. Validate Key Columns
- Check that columns used for plotting actually exist
- Verify columns have data (not all null/NaN)
- Check for duplicate column names (e.g., `concurrency` and `concurrency.1`)
- Confirm data types are correct (numeric for plotting)

### 3. Check Data Ranges
```python
# For columns you'll plot:
print(df['column_name'].describe())
print(df['column_name'].unique())
```

## After Creating Visualizations

### 1. Visual Inspection Required
**Always open and inspect generated images to verify:**
- [ ] Data points are visible (not all missing)
- [ ] Axes have appropriate labels
- [ ] Legend is readable and not obscuring data
- [ ] Colors follow the palette standard (check skills/visualization-palette)
- [ ] No text overlays obscuring important details
- [ ] Grid lines don't make bars illegible
- [ ] X-axis labels are readable (not too crowded)

### 2. Regeneration Test
If you modify data or scripts:
```bash
# Always regenerate visualizations after data changes
python3 visualize_*.py
```

## Common Pitfalls

### Issue: Missing Data Points in Graphs
**Cause**: Using wrong column name or null data
**Fix**:
```python
# Check column before using
if 'concurrency' in df.columns:
    print(df['concurrency'].isna().sum(), "nulls")
# Look for alternative column names
print([c for c in df.columns if 'concurr' in c.lower()])
```

### Issue: Graph Shows No Lines/Bars
**Cause**: DataFrame filtering removed all data
**Fix**:
```python
# Verify filtered data exists
filtered = df[df['model'] == 'Qwen3-8B']
print(f"Filtered rows: {len(filtered)}")
assert len(filtered) > 0, "No data after filtering!"
```

### Issue: Overlapping/Illegible Text
**Cause**: Too many x-axis labels or text annotations
**Fix**:
- Reduce tick frequency: `ax.set_xticks(x[::2])` (every other)
- Rotate labels: `rotation=45`
- Remove summary text boxes from data plots

### Issue: Wrong Colors
**Cause**: Not using palette from skills/visualization-palette
**Fix**:
```python
import seaborn as sns
sns.set_style("whitegrid")
sns.set_palette("muted")
palette_colors = sns.color_palette("muted")
colors = {'config1': palette_colors[0], 'config2': palette_colors[1]}
```

## Validation Script Template

Add this to analysis scripts:
```python
def validate_dataframe(df, required_cols, name="DataFrame"):
    """Validate DataFrame has expected structure."""
    print(f"\n=== Validating {name} ===")
    print(f"Shape: {df.shape}")

    # Check required columns exist
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Check for null values
    for col in required_cols:
        null_count = df[col].isna().sum()
        if null_count > 0:
            print(f"WARNING: {col} has {null_count} null values")

    # Check for data
    if len(df) == 0:
        raise ValueError(f"{name} is empty!")

    print(f"✓ {name} validated")
    return True

# Use it:
validate_dataframe(df, ['model', 'concurrency', 'throughput'], "Benchmark Results")
```

## Pre-Commit Checklist

Before committing visualizations:
- [ ] All graphs visually inspected
- [ ] Data points visible on all plots
- [ ] Colors match palette standards
- [ ] No text obscuring data
- [ ] Axes labeled correctly
- [ ] No duplicate/test plots included
