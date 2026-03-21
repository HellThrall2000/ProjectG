import timeit
import pandas as pd
from langchain_core.documents import Document

def get_speaker(transliteration):
    if type(transliteration) == str and "uvāca" in transliteration:
        return transliteration.split(" uvāca")[0].strip()
    return "Unknown"

df = pd.read_csv("data/Bhagwad_Gita.csv")
df["Speaker"] = df["Transliteration"].apply(get_speaker)
krishna_df = df[df["Speaker"] == "śrī-bhagavān"]

def run_iterrows():
    documents = []
    for _, row in krishna_df.iterrows():
        metadata = {
            "chapter": row["Chapter"],
            "verse": row["Verse"],
            "speaker": "Krishna"
        }
        document = Document(page_content=row["EngMeaning"], metadata=metadata)
        documents.append(document)
    return documents

def run_itertuples():
    documents = []
    for row in krishna_df.itertuples():
        metadata = {
            "chapter": row.Chapter,
            "verse": row.Verse,
            "speaker": "Krishna"
        }
        document = Document(page_content=row.EngMeaning, metadata=metadata)
        documents.append(document)
    return documents

if __name__ == '__main__':
    n = 100
    time_iterrows = timeit.timeit(run_iterrows, number=n)
    time_itertuples = timeit.timeit(run_itertuples, number=n)

    docs_iter = run_iterrows()
    docs_tuple = run_itertuples()

    print(f"Number of documents: {len(docs_iter)}")
    print(f"Output identical: {docs_iter == docs_tuple}")
    print(f"iterrows() time for {n} runs: {time_iterrows:.4f} seconds")
    print(f"itertuples() time for {n} runs: {time_itertuples:.4f} seconds")
    print(f"Speedup: {time_iterrows / time_itertuples:.2f}x")
