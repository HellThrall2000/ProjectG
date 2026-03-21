import pandas as pd
import time
from langchain_core.documents import Document

def benchmark():
    # Create a dummy dataframe
    data = {
        "Chapter": [i for i in range(10000)],
        "Verse": [i for i in range(10000)],
        "EngMeaning": [f"Meaning {i}" for i in range(10000)]
    }
    df = pd.DataFrame(data)

    # Benchmark iterrows
    start = time.time()
    documents = []
    for _, row in df.iterrows():
        metadata = {
            "chapter": row["Chapter"],
            "verse": row["Verse"],
            "speaker": "Krishna"
        }
        document = Document(page_content=row["EngMeaning"], metadata=metadata)
        documents.append(document)
    end = time.time()
    iterrows_time = end - start
    print(f"iterrows time: {iterrows_time:.4f}s")

    # Benchmark itertuples
    start = time.time()
    documents = []
    for row in df.itertuples(index=False):
        metadata = {
            "chapter": row.Chapter,
            "verse": row.Verse,
            "speaker": "Krishna"
        }
        document = Document(page_content=row.EngMeaning, metadata=metadata)
        documents.append(document)
    end = time.time()
    itertuples_time = end - start
    print(f"itertuples time: {itertuples_time:.4f}s")

    print(f"Speedup: {iterrows_time / itertuples_time:.2f}x")

if __name__ == "__main__":
    benchmark()