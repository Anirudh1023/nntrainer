"""
Convert LaMP-3 reviews for a single user into nntrainer chat-format training/test files.
User 206515: 113 reviews, balanced distribution 1:10 2:13 3:30 4:23 5:37
"""
import random
from collections import Counter
from datasets import load_dataset

USER_ID = "206515"
SEED = 42
TRAIN_RATIO = 0.8
MAX_SOURCE_CHARS = 400
TRAIN_OUT = "lamp3_user_train.txt"
TEST_OUT  = "lamp3_user_test.txt"

random.seed(SEED)

print("Loading dataset...")
ds = load_dataset("haotiansun014/LaMP")
train_ds = ds["train"]

# Collect unique reviews for this user
seen_src = {}
for s in train_ds:
    if s["id"] == USER_ID and s["target"] in {"1","2","3","4","5"}:
        src = s["source"]
        if src not in seen_src:
            seen_src[src] = s

samples = list(seen_src.values())
random.shuffle(samples)
print(f"User {USER_ID}: {len(samples)} unique reviews")
print("Distribution:", dict(sorted(Counter(s["target"] for s in samples).items())))

split = int(len(samples) * TRAIN_RATIO)
train_samples = samples[:split]
test_samples  = samples[split:]
print(f"Train: {len(train_samples)}  Test: {len(test_samples)}")

def write_chat_file(path, samples):
    with open(path, "w") as f:
        for s in samples:
            # Strip the boilerplate prefix, keep just the review text
            src = s["source"]
            if "review:" in src:
                review = src.split("review:", 1)[1].strip()[:MAX_SOURCE_CHARS]
                source = (
                    "What is the score of the following review on a scale of 1 to 5? "
                    "Just answer with 1, 2, 3, 4, or 5 without further explanation. "
                    f"review: {review}"
                )
            else:
                source = src[:MAX_SOURCE_CHARS]
            f.write(f"<|im_start|>user\n{source}<|im_end|>\n<|im_start|>assistant\n{s['target']}\n")

write_chat_file(TRAIN_OUT, train_samples)
write_chat_file(TEST_OUT,  test_samples)
print(f"Written: {TRAIN_OUT}  {TEST_OUT}")

# Print a few examples
print("\n--- Sample training entries ---")
with open(TRAIN_OUT) as f:
    lines = f.readlines()
for i in range(0, min(8, len(lines)), 4):
    print(f"rating={lines[i+3].strip()}  review={lines[i+1][100:180].strip()}")
