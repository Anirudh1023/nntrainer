"""Convert LaMP-3 (product rating 1-5) to nntrainer chat-format training file."""
import random
from datasets import load_dataset

SEED = 42
MAX_SAMPLES = 500
MAX_SOURCE_CHARS = 400
OUT_FILE = "lamp3_train.txt"

random.seed(SEED)

print("Loading dataset...")
ds = load_dataset("haotiansun014/LaMP")

# Filter LaMP-3: rating 1-5
lamp3 = [s for s in ds["train"] if s["target"] in {"1", "2", "3", "4", "5"}]
print(f"LaMP-3 total: {len(lamp3)}")

# Balanced sample across ratings
from collections import defaultdict
by_rating = defaultdict(list)
for s in lamp3:
    by_rating[s["target"]].append(s)

per_class = MAX_SAMPLES // 5
samples = []
for rating in ["1", "2", "3", "4", "5"]:
    pool = by_rating[rating]
    random.shuffle(pool)
    samples.extend(pool[:per_class])

random.shuffle(samples)
print(f"Sampled: {len(samples)} ({per_class} per rating)")

# Write chat-format file
# Each sample is a 4-line block: <|im_start|>user / source<|im_end|> / <|im_start|>assistant / rating
# loadTextFile groups lines until next <|im_start|>user → correct single sample
# dataCb trains: predict rating token given full chat context
with open(OUT_FILE, "w") as f:
    for s in samples:
        source = s["source"][:MAX_SOURCE_CHARS].strip()
        target = s["target"]
        f.write(f"<|im_start|>user\n{source}<|im_end|>\n<|im_start|>assistant\n{target}\n")

print(f"Written to {OUT_FILE}")
print("Sample:")
with open(OUT_FILE) as f:
    for i, line in enumerate(f):
        print(line, end="")
        if i >= 3:
            print("...")
            break
