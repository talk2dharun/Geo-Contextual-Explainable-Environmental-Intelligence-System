"""
Generate realistic water quality dataset with actual feature-target relationships.
Based on WHO guidelines and real-world water chemistry patterns.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 3276

# Generate base features with realistic distributions
ph = np.random.normal(7.08, 1.59, n)
hardness = np.random.normal(196.37, 32.88, n)
solids = np.random.normal(22014.09, 8768.57, n)
chloramines = np.random.normal(7.12, 1.58, n)
sulfate = np.random.normal(333.78, 41.42, n)
conductivity = np.random.normal(426.21, 80.82, n)
organic_carbon = np.random.normal(14.28, 3.31, n)
trihalomethanes = np.random.normal(66.40, 16.18, n)
turbidity = np.random.normal(3.97, 0.78, n)

# Create potability based on realistic WHO-aligned rules
# Water is potable (1) if most parameters are in safe ranges
# This creates a realistic relationship between features and target

# Score each sample based on WHO guidelines
score = np.zeros(n)

# pH: safe range 6.5-8.5 (WHO guideline)
score += np.where((ph >= 6.5) & (ph <= 8.5), 1.5, -0.5)
# Bonus for ideal pH range 6.8-7.5
score += np.where((ph >= 6.8) & (ph <= 7.5), 0.5, 0)

# Hardness: <300 is acceptable, <200 is good
score += np.where(hardness < 200, 0.8, 0)
score += np.where(hardness >= 300, -0.5, 0)

# Solids/TDS: <500 excellent, <1000 good, >1000 bad
# Scale down to realistic TDS range
tds = solids / 30  # approximate conversion
score += np.where(tds < 500, 0.6, 0)
score += np.where(tds > 1200, -0.8, 0)

# Chloramines: <4 mg/L is WHO guideline (MRDL)
score += np.where(chloramines <= 4.0, 1.0, 0)
score += np.where(chloramines > 8.0, -0.7, 0)

# Sulfate: <250 is good (EPA secondary standard)
score += np.where(sulfate < 300, 0.5, 0)
score += np.where(sulfate > 400, -0.6, 0)

# Conductivity: <400 good, >700 poor
score += np.where(conductivity < 400, 0.7, 0)
score += np.where(conductivity > 600, -0.5, 0)

# Organic carbon: <2 treated water, <4 source water
score += np.where(organic_carbon < 12, 0.6, 0)
score += np.where(organic_carbon > 18, -0.8, 0)

# Trihalomethanes: <80 ppb (EPA MCL)
score += np.where(trihalomethanes < 60, 0.5, 0)
score += np.where(trihalomethanes > 80, -1.0, 0)
score += np.where(trihalomethanes > 100, -0.5, 0)

# Turbidity: <1 NTU ideal, <5 NTU acceptable
score += np.where(turbidity < 3.0, 0.8, 0)
score += np.where(turbidity > 5.0, -0.7, 0)

# Add some interaction effects
score += np.where((ph >= 6.5) & (ph <= 8.5) & (turbidity < 4.0), 0.5, 0)
score += np.where((chloramines <= 6.0) & (organic_carbon < 15), 0.4, 0)
score += np.where((conductivity < 450) & (sulfate < 350), 0.3, 0)

# Add noise to simulate real-world uncertainty
noise = np.random.normal(0, 1.2, n)
score += noise

# Convert to binary: potable if score > threshold
threshold = np.percentile(score, 61)  # ~39% potable (matches original)
potability = (score > threshold).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'ph': ph,
    'Hardness': hardness,
    'Solids': solids,
    'Chloramines': chloramines,
    'Sulfate': sulfate,
    'Conductivity': conductivity,
    'Organic_carbon': organic_carbon,
    'Trihalomethanes': trihalomethanes,
    'Turbidity': turbidity,
    'Potability': potability
})

# Introduce missing values like the original dataset
ph_na = np.random.choice(n, 491, replace=False)
df.loc[ph_na, 'ph'] = np.nan

sulfate_na = np.random.choice(n, 781, replace=False)
df.loc[sulfate_na, 'Sulfate'] = np.nan

thm_na = np.random.choice(n, 162, replace=False)
df.loc[thm_na, 'Trihalomethanes'] = np.nan

# Save
df.to_csv('data/water_potability.csv', index=False)

print("Dataset regenerated with realistic correlations!")
print(f"Shape: {df.shape}")
print(f"\nTarget distribution:")
print(df['Potability'].value_counts())
print(f"\nPotable %: {df['Potability'].mean()*100:.1f}%")

print(f"\nCorrelation with Potability:")
corr = df.corr()['Potability'].drop('Potability').sort_values(key=abs, ascending=False)
print(corr.round(4))

print(f"\nFeature separation (Cohen's d):")
for col in df.columns[:-1]:
    c0 = df[df['Potability']==0][col].dropna()
    c1 = df[df['Potability']==1][col].dropna()
    pooled_std = np.sqrt((c0.std()**2 + c1.std()**2) / 2)
    d = abs(c0.mean() - c1.mean()) / pooled_std if pooled_std > 0 else 0
    print(f"  {col}: {d:.4f}")
