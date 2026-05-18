
import ogbench
from ogbench.utils import download_datasets

print("Starting dataset download...")
download_datasets([
    'humanoidmaze-medium-navigate-v0',
    'cube-triple-play-v0'
])
print("Download complete!")
