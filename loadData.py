import matplotlib.pyplot as plt
from datasets import load_dataset, Image

#ds = load_dataset("keremberke/pokemon-classification", name="full", split="train[100:200]")

ds = load_dataset("keremberke/pokemon-classification", name="full")
example = ds['train'][0]

#example = ds[0]["image"]
plt.imshow(example)
plt.show()
#example = ds['train'][0]
