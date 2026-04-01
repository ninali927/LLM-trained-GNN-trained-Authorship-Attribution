import time
import pandas as pd

from build_dataset import build_dataset
from WAN.WAN_pipeline import WAN_distance_pipeline


def build_author_label_map(df):
    """
    Create a mapping from author name to integer label.
    """
    authors = sorted(df["author"].unique())
    author_to_label = {}

    for i in range(len(authors)):
        author_to_label[authors[i]] = i

    return author_to_label


def distance_to_similarity(distance):
    """
    Convert WAN distance to similarity weight for graph edges.
    """
    return 1.0 / (1.0 + distance)


def create_nodes_dataframe(df):
    """
    Create node dataframe from chunked dataset.
    Each chunk becomes one node.
    """
    author_to_label = build_author_label_map(df)

    rows = []

    for i in range(len(df)):
        row = df.iloc[i]

        rows.append({
            "node_id": i,
            "chunk_id": row["chunk_id"],
            "text": row["text"],
            "author": row["author"],
            "author_label": author_to_label[row["author"]],
            "play": row["play"],
            "chunk_index": row["chunk_index"],
            "source_file": row["source_file"],
            "num_words": row["num_words"]
        })

    nodes_df = pd.DataFrame(rows)
    return nodes_df, author_to_label


def build_edges_pairwise(nodes_df,
                               function_words,
                               D=10,
                               alpha=0.75,
                               epsilon=1e-12,
                               distance_type="kl"):
    """
    Build all-pairs graph edges with detailed progress printing.
    """
    edge_rows = []
    n = len(nodes_df)

    total_pairs = n * (n - 1) // 2
    pair_count = 0

    print("\n[Edge Step] Starting all-pairs edge construction...")
    print("Number of nodes:", n)
    print("Total undirected pairs to process:", total_pairs)

    all_edges_start = time.time()

    for i in range(n):
        text_i = nodes_df.iloc[i]["text"]
        print(f"\n--- Outer loop i = {i}/{n-1} ---")

        for j in range(i + 1, n):
            text_j = nodes_df.iloc[j]["text"]
            pair_count += 1

            print(f"Processing pair {pair_count}/{total_pairs}: (node {i}, node {j})")
            pair_start = time.time()

            try:
                distance = WAN_distance_pipeline(
                    chunk_text_1=text_i,
                    chunk_text_2=text_j,
                    function_words=function_words,
                    D=D,
                    alpha=alpha,
                    epsilon=epsilon,
                    distance_type=distance_type
                )

                weight = distance_to_similarity(distance)

                edge_rows.append({
                    "source": i,
                    "target": j,
                    "distance": distance,
                    "weight": weight
                })

                edge_rows.append({
                    "source": j,
                    "target": i,
                    "distance": distance,
                    "weight": weight
                })

                pair_end = time.time()
                print(f"Finished pair ({i}, {j}) | distance = {distance:.6f} | time = {pair_end - pair_start:.2f} sec")

            except Exception as e:
                pair_end = time.time()
                print(f"Error on pair ({i}, {j}) after {pair_end - pair_start:.2f} sec")
                print(e)

    all_edges_end = time.time()

    print("\n[Edge Step] Finished all-pairs edge construction.")
    print("Total edge-building time:", round(all_edges_end - all_edges_start, 2), "seconds")
    print("Number of directed edges created:", len(edge_rows))

    edges_df = pd.DataFrame(edge_rows)
    return edges_df


def graph_construction_pairwise(input_folder,
                            chunked_dataset_file,
                            nodes_output_file,
                            edges_output_file,
                            function_words,
                            chunk_size=5000,
                            D=10,
                            alpha=0.75,
                            epsilon=1e-12,
                            distance_type="kl"):
    """
    Full graph construction test pipeline with detailed timing.
    """

    total_start = time.time()

    # -----------------------------------
    # Step 1: Build chunked dataset
    # -----------------------------------
    print("\n========== STEP 1: BUILD DATASET ==========")
    step1_start = time.time()

    build_dataset(
        input_folder=input_folder,
        output_file=chunked_dataset_file,
        chunk_size=chunk_size
    )

    step1_end = time.time()
    print("Finished Step 1.")
    print("Chunked dataset file:", chunked_dataset_file)
    print("Step 1 time:", round(step1_end - step1_start, 2), "seconds")

    # -----------------------------------
    # Step 2: Read CSV
    # -----------------------------------
    print("\n========== STEP 2: READ DATASET CSV ==========")
    step2_start = time.time()

    df = pd.read_csv(chunked_dataset_file)

    step2_end = time.time()
    print("Finished Step 2.")
    print("Number of dataset rows:", len(df))
    print("Step 2 time:", round(step2_end - step2_start, 2), "seconds")

    # -----------------------------------
    # Step 3: Create nodes
    # -----------------------------------
    print("\n========== STEP 3: CREATE NODES ==========")
    step3_start = time.time()

    nodes_df, author_to_label = create_nodes_dataframe(df)

    step3_end = time.time()
    print("Finished Step 3.")
    print("Number of nodes:", len(nodes_df))
    print("Author to label mapping:", author_to_label)
    print("Step 3 time:", round(step3_end - step3_start, 2), "seconds")

    # -----------------------------------
    # Step 4: Build edges
    # -----------------------------------
    print("\n========== STEP 4: BUILD EDGES ==========")
    step4_start = time.time()

    edges_df = build_edges_pairwise(
        nodes_df=nodes_df,
        function_words=function_words,
        D=D,
        alpha=alpha,
        epsilon=epsilon,
        distance_type=distance_type
    )

    step4_end = time.time()
    print("Finished Step 4.")
    print("Number of edges:", len(edges_df))
    print("Step 4 time:", round(step4_end - step4_start, 2), "seconds")

    # -----------------------------------
    # Step 5: Save outputs
    # -----------------------------------
    print("\n========== STEP 5: SAVE OUTPUTS ==========")
    step5_start = time.time()

    nodes_df.to_csv(nodes_output_file, index=False)
    edges_df.to_csv(edges_output_file, index=False)

    step5_end = time.time()
    print("Finished Step 5.")
    print("Nodes saved to:", nodes_output_file)
    print("Edges saved to:", edges_output_file)
    print("Step 5 time:", round(step5_end - step5_start, 2), "seconds")

    total_end = time.time()

    print("\n========== GRAPH CONSTRUCTION (PAIRWISE) FINISHED ==========")
    print("Final number of nodes:", len(nodes_df))
    print("Final number of edges:", len(edges_df))
    print("Total time:", round(total_end - total_start, 2), "seconds")


if __name__ == "__main__":
    input_folder = "data/test_plays"
    chunked_dataset_file = "data/chunked_plays.csv"
    nodes_output_file = "data/graph_nodes.csv"
    edges_output_file = "data/graph_edges.csv"

    function_words = [
        "the", "a", "an", "and", "or", "but", "to", "of", "in", "on",
        "for", "with", "as", "at", "by", "from", "that", "this", "it",
        "he", "she", "i", "you", "we", "they", "is", "was", "be", "been",
        "are", "were", "not", "do", "does", "did", "have", "has", "had"
    ]

    graph_construction_pairwise(
        input_folder=input_folder,
        chunked_dataset_file=chunked_dataset_file,
        nodes_output_file=nodes_output_file,
        edges_output_file=edges_output_file,
        function_words=function_words,
        chunk_size=5000,
        D=10,
        alpha=0.75,
        epsilon=1e-12,
        distance_type="kl"
    )