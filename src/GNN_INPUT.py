import pandas as pd
import torch

from GRAPH_CONSTRUCTION_PAIRWISE import graph_construction_pairwise
from GRAPH_CONSTRUCTION_ANNOY import graph_construction_annoy


def load_texts_and_graph(nodes_csv, edges_csv, chunked_dataset_file=None):
    """
    Load graph structure and node texts.
    Used for joint LLM + GNN training, where embeddings are computed
    dynamically during the forward pass.

    Inputs:
    - nodes_csv: path to graph nodes CSV (must contain 'author_label')
    - edges_csv: path to graph edges CSV (must contain 'source', 'target', 'weight')
    - chunked_dataset_file: optional path to chunked plays CSV (must contain 'text').
        If provided, texts are loaded from here.
        If None, texts are loaded from the 'text' column of nodes_csv.

    Returns:
    - texts: list of strings, one per node
    - edge_index: [2, num_edges] LongTensor
    - edge_weight: [num_edges] FloatTensor
    - y: [num_nodes] LongTensor of author labels
    """
    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)

    num_nodes = len(nodes_df)

    if chunked_dataset_file is not None:
        chunks_df = pd.read_csv(chunked_dataset_file)
        texts = chunks_df["text"].fillna("").tolist()
    else:
        if "text" not in nodes_df.columns:
            raise ValueError(
                "nodes_csv does not have a 'text' column. "
                "Please provide chunked_dataset_file."
            )
        texts = nodes_df["text"].fillna("").tolist()

    if len(texts) != num_nodes:
        raise ValueError(
            f"Number of texts ({len(texts)}) does not match "
            f"number of nodes ({num_nodes})."
        )

    y = torch.tensor(nodes_df["author_label"].values, dtype=torch.long)

    edge_index = torch.tensor(
        edges_df[["source", "target"]].values.T,
        dtype=torch.long
    )

    edge_weight = torch.tensor(
        edges_df["weight"].values,
        dtype=torch.float
    )

    return texts, edge_index, edge_weight, y


def prepare_gnn_inputs(
    input_folder,
    chunked_dataset_file,
    nodes_csv,
    edges_csv,
    function_words,
    chunk_size=5000,
    D=10,
    alpha=0.75,
    epsilon=1e-12,
    distance_type="kl",
    graph_mode="annoy",
    k=10,
    num_trees=20,
    search_k=-1,
    renyi_alpha=0.5
):
    """
    Build the graph and return texts + graph tensors ready for joint LLM+GNN training.
    Embedding generation is removed — embeddings are now computed dynamically
    during the forward pass.

    graph_mode:
        - "pairwise": dense graph from pairwise distances
        - "annoy": sparse graph from Annoy candidate retrieval

    Returns:
    - texts: list of strings, one per node
    - edge_index: [2, num_edges] LongTensor
    - edge_weight: [num_edges] FloatTensor
    - y: [num_nodes] LongTensor of author labels
    """

    # -----------------------------------
    # 1. Build graph
    # -----------------------------------
    print("\n[1] Building graph...")

    if graph_mode == "pairwise":
        graph_construction_pairwise(
            input_folder=input_folder,
            chunked_dataset_file=chunked_dataset_file,
            nodes_output_file=nodes_csv,
            edges_output_file=edges_csv,
            function_words=function_words,
            chunk_size=chunk_size,
            D=D,
            alpha=alpha,
            epsilon=epsilon,
            distance_type=distance_type
        )

    elif graph_mode == "annoy":
        graph_construction_annoy(
            input_folder=input_folder,
            chunked_dataset_file=chunked_dataset_file,
            nodes_output_file=nodes_csv,
            edges_output_file=edges_csv,
            function_words=function_words,
            chunk_size=chunk_size,
            D=D,
            alpha=alpha,
            epsilon=epsilon,
            distance_type=distance_type,
            k=k,
            num_trees=num_trees,
            search_k=search_k,
            renyi_alpha=renyi_alpha
        )

    else:
        raise ValueError("graph_mode must be 'pairwise' or 'annoy'")

    # -----------------------------------
    # 2. Load texts + graph tensors
    # -----------------------------------
    print("\n[2] Loading texts and graph tensors...")
    texts, edge_index, edge_weight, y = load_texts_and_graph(
        nodes_csv=nodes_csv,
        edges_csv=edges_csv,
        chunked_dataset_file=chunked_dataset_file
    )

    return texts, edge_index, edge_weight, y


if __name__ == "__main__":
    input_folder         = "data/test_plays"
    chunked_dataset_file = "data/chunked_plays.csv"
    nodes_csv            = "data/graph_nodes.csv"
    edges_csv            = "data/graph_edges.csv"

    function_words = [
        "the", "a", "an", "and", "or", "but", "to", "of", "in", "on",
        "for", "with", "as", "at", "by", "from", "that", "this", "it",
        "he", "she", "i", "you", "we", "they", "is", "was", "be", "been",
        "are", "were", "not", "do", "does", "did", "have", "has", "had"
    ]

    texts, edge_index, edge_weight, y = prepare_gnn_inputs(
        input_folder=input_folder,
        chunked_dataset_file=chunked_dataset_file,
        nodes_csv=nodes_csv,
        edges_csv=edges_csv,
        function_words=function_words,
        chunk_size=5000,
        D=10,
        alpha=0.75,
        epsilon=1e-12,
        distance_type="bhattacharyya",
        graph_mode="annoy",
        k=3,
        num_trees=20,
        search_k=-1,
        renyi_alpha=0.5
    )

    print("\nFinished preparing GNN inputs.")
    print("Nodes:", len(texts))
    print("edge_index shape:", edge_index.shape)
    print("edge_weight shape:", edge_weight.shape)
    print("y shape:", y.shape)