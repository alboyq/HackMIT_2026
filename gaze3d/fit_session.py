import json
import os
from gaze3d.pipeline import Pipeline, PROFILES

p = Pipeline(source='push')
p.load()

# Load the latest session
session = 'session-20260920-143520.jsonl'
samples = []
with open(os.path.join(PROFILES, session)) as f:
    for line in f:
        data = json.loads(line)
        # Skip metadata lines, only include actual samples with 'o' field
        if 'o' in data:
            samples.append(data)

print(f'Loaded {len(samples)} samples')

# Set samples and fit
p.samples = samples
report = p.fit()
print(f'Fit done! CV error: {report["cv_err_px"]:.1f}px')

# Save as default
path = p.save_profile('default')
print(f'Saved to {path}')
