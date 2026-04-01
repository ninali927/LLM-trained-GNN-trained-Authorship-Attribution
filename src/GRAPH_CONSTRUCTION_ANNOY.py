import time
import numpy as np
import pandas as pd
from annoy import AnnoyIndex

from build_dataset import build_dataset
from WAN.WAN_pipeline import WAN_distance_pipeline
from preprocess.preprocess_pipeline import preprocess_chunk_text
from WAN.wan_matrix import build_wan_from_sentences
from WAN.markov_normalization import markov_normalization

from WAN.relative_entropy.Bhattacharyya_Distance import get_bhattacharyya_annoy_vector
from WAN.relative_entropy.Hellinger_Distance import get_hellinger_annoy_vector
from WAN.relative_entropy.Kullback_Leibler_Divergence import get_kl_annoy_vector
from WAN.relative_entropy.Renyi_Divergence import get_renyi_annoy_vector


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


def build_annoy_vector_from_chunk_text(chunk_text,
                                       function_words,
                                       D=10,
                                       alpha=0.75,
                                       distance_type="bhattacharyya",
                                       renyi_alpha=0.5):
    """
    Build one Annoy vector directly from one chunk of text.

    Logic:
    chunk_text
        -> preprocess
        -> WAN adjacency matrix A
        -> Markov matrix P
        -> Annoy vector
    """
    _, _, _, sentences = preprocess_chunk_text(chunk_text)

    A = build_wan_from_sentences(
        sentences,
        function_words=function_words,
        D=D,
        alpha=alpha
    )

    P = markov_normalization(A)

    distance_type = distance_type.lower()

    if distance_type == "bhattacharyya":
        v = get_bhattacharyya_annoy_vector(P)

    elif distance_type == "hellinger":
        v = get_hellinger_annoy_vector(P)

    elif distance_type == "kl":
        v = get_kl_annoy_vector(P, role="query")

    elif distance_type == "renyi":
        v = get_renyi_annoy_vector(P, alpha=renyi_alpha, role="query")

    else:
        raise ValueError(
            "For Annoy graph construction, distance_type must be one of: "
            "'bhattacharyya', 'hellinger', 'kl', 'renyi'"
        )

    v = np.asarray(v, dtype=np.float32)

    norm = np.linalg.norm(v)

    if norm < 1e-8:
        raise ValueError("Annoy vector has near-zero norm")

    v = v / norm

    return v


def build_edges_annoy(nodes_df,
                      function_words,
                      D=10,
                      alpha=0.75,
                      epsilon=1e-12,
                      distance_type="bhattacharyya",
                      k=10,
                      num_trees=20,
                      search_k=-1,
                      renyi_alpha=0.5,
                      include_self=False):
    """
    Build sparse directed graph edges using Annoy.

    Logic:
    1. For each node, build an Annoy vector from its text
    2. Build Annoy index
    3. For each node i, retrieve k neighbors
    4. Only for those neighbors, compute exact WAN distance
       using the existing WAN_distance_pipeline
    """
    edge_rows = []
    n = len(nodes_df)

    print("\n[Edge Step] Starting sparse edge construction...")
    print("Number of nodes:", n)
    print("k nearest neighbors per node:", k)

    all_edges_start = time.time()

    # -----------------------------------
    # Step A: Build Annoy vectors
    # -----------------------------------
    print("\n[Edge Step A] Building Annoy vectors...")
    stepA_start = time.time()

    vectors = []

    for i in range(n):
        text_i = nodes_df.iloc[i]["text"]
        print(f"Building Annoy vector for node {i}/{n-1}")

        v = build_annoy_vector_from_chunk_text(
            chunk_text=text_i,
            function_words=function_words,
            D=D,
            alpha=alpha,
            distance_type=distance_type,
            renyi_alpha=renyi_alpha
        )

        vectors.append(np.asarray(v, dtype=np.float32))

    dim = len(vectors[0])

    for i in range(len(vectors)):
        if len(vectors[i]) != dim:
            raise ValueError("All Annoy vectors must have the same dimension")

    stepA_end = time.time()
    print("Finished Step A.")
    print("Annoy vector dimension:", dim)
    print("Step A time:", round(stepA_end - stepA_start, 2), "seconds")

    # -----------------------------------
    # Step B: Build Annoy index
    # -----------------------------------
    print("\n[Edge Step B] Building Annoy index...")
    stepB_start = time.time()

    index = AnnoyIndex(dim, "angular")

    for i in range(n):
        index.add_item(i, vectors[i])

    index.build(num_trees)

    stepB_end = time.time()
    print("Finished Step B.")
    print("Step B time:", round(stepB_end - stepB_start, 2), "seconds")
    
    print("Number of items in Annoy index:", index.get_n_items())

    print("Test query item 0:", index.get_nns_by_item(0, 11))
    print("Test query item 1:", index.get_nns_by_item(1, 11))
    print("Test query item 2:", index.get_nns_by_item(2, 11))

    # -----------------------------------
    # Step C: Query neighbors and compute exact distances
    # -----------------------------------
    print("\n[Edge Step C] Querying neighbors and computing exact WAN distances...")
    stepC_start = time.time()

    for i in range(n):
        text_i = nodes_df.iloc[i]["text"]

        print(f"\n--- Query node i = {i}/{n-1} ---")

        query_vector = vectors[i]
        neighbors = index.get_nns_by_vector(query_vector, k + 1, search_k=search_k)
        print(f"Retrieved neighbors for node {i}: {neighbors}")

        for j in neighbors:
            if (not include_self) and (j == i):
                continue

            text_j = nodes_df.iloc[j]["text"]
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

                pair_end = time.time()
                print(
                    f"Finished edge ({i} -> {j}) | "
                    f"distance = {distance:.6f} | "
                    f"time = {pair_end - pair_start:.2f} sec"
                )

            except Exception as e:
                pair_end = time.time()
                print(f"Error on edge ({i} -> {j}) after {pair_end - pair_start:.2f} sec")
                print(e)

    stepC_end = time.time()

    print("\n[Edge Step] Finished sparse edge construction.")
    print("Step C time:", round(stepC_end - stepC_start, 2), "seconds")
    print("Total edge-building time:", round(stepC_end - all_edges_start, 2), "seconds")
    print("Number of directed edges created:", len(edge_rows))

    edges_df = pd.DataFrame(edge_rows)
    return edges_df


def graph_construction_annoy(input_folder,
                             chunked_dataset_file,
                             nodes_output_file,
                             edges_output_file,
                             function_words,
                             chunk_size=5000,
                             D=10,
                             alpha=0.75,
                             epsilon=1e-12,
                             distance_type="bhattacharyya",
                             k=10,
                             num_trees=20,
                             search_k=-1,
                             renyi_alpha=0.5):
    """
    Full graph construction sparse pipeline with detailed timing.
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
    # Step 4: Build sparse edges
    # -----------------------------------
    print("\n========== STEP 4: BUILD SPARSE EDGES ==========")
    step4_start = time.time()

    edges_df = build_edges_annoy(
        nodes_df=nodes_df,
        function_words=function_words,
        D=D,
        alpha=alpha,
        epsilon=epsilon,
        distance_type=distance_type,
        k=k,
        num_trees=num_trees,
        search_k=search_k,
        renyi_alpha=renyi_alpha,
        include_self=False
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

    print("\n========== GRAPH CONSTRUCTION (ANNOY / SPARSE) FINISHED ==========")
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

    graph_construction_annoy(
        input_folder=input_folder,
        chunked_dataset_file=chunked_dataset_file,
        nodes_output_file=nodes_output_file,
        edges_output_file=edges_output_file,
        function_words=function_words,
        chunk_size=5000,
        D=10,
        alpha=0.75,
        epsilon=1e-12,
        distance_type="bhattacharyya",
        k=3,
        num_trees=20,
        search_k=-1,
        renyi_alpha=0.5
    )