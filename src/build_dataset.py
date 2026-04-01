import os
import pandas as pd


def build_dataset(input_folder, output_file, chunk_size=5000):
    """
    Read raw play text files and create a chunked dataset.

    Each row of the output dataset contains:
        chunk_id    : unique id for the chunk
        text        : chunk of the play
        author      : author name
        play        : play title
        chunk_index : index of chunk within the play
        source_file : original txt filename
        num_words   : number of words in this chunk
    """

    rows = []

    # loop through files
    for fname in sorted(os.listdir(input_folder)):
        if not fname.endswith(".txt"):
            continue

        # extract author
        author = fname.split("_")[0]

        # extract play name
        play = fname.replace(".txt", "").split("_", 1)[1]
        path = os.path.join(input_folder, fname)

        # read file
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        words = text.split()

        # chunk text
        for chunk_index, start in enumerate(range(0, len(words), chunk_size)):
            chunk_words = words[start:start + chunk_size]
            chunk_text = " ".join(chunk_words)

            chunk_id = f"{author}_{play}_{chunk_index}"

            rows.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "author": author,
                "play": play,
                "chunk_index": chunk_index,
                "source_file": fname,
                "num_words": len(chunk_words)
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)

    print("Dataset created:", output_file)
    print("Number of rows:", len(df))


if __name__ == "__main__":
    input_folder = "data/test_plays"
    output_file = "data/chunked_plays.csv"

    build_dataset(input_folder, output_file)