import pandas as pd
import time

# Create dummy data
data = {"Chapter": range(10000), "Verse": range(10000), "EngMeaning": ["meaning"] * 10000}
df = pd.DataFrame(data)

start = time.time()
for _, row in df.iterrows():
    c = row["Chapter"]
    v = row["Verse"]
    m = row["EngMeaning"]
end = time.time()
print(f"iterrows: {end - start:.4f} seconds")

start = time.time()
for row in df.itertuples(index=False):
    c = row.Chapter
    v = row.Verse
    m = row.EngMeaning
end = time.time()
print(f"itertuples: {end - start:.4f} seconds")
