"""Verify EuroSAT dataset and start CNN training."""
from pathlib import Path

eurosat_dir = Path(r"d:\Spatio-Temporal UHI prediction model\data\raw\EuroSAT_RGB")
classes = sorted([d.name for d in eurosat_dir.iterdir() if d.is_dir()])
print(f"Classes found: {len(classes)}")
total = 0
for c in classes:
    count = len(list((eurosat_dir / c).glob("*.*")))
    total += count
    print(f"  {c}: {count} images")
print(f"\nTotal: {total} images")
