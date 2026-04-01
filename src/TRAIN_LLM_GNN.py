import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from GNN_MODELS import GCN, SAGE, GIN, GAT
from GNN_INPUT import prepare_gnn_inputs




def mean_pooling(last_hidden_state, attention_mask):
    """
    Mean-pool token embeddings, ignoring padding tokens.
    """
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_hidden = last_hidden_state * mask
    summed = masked_hidden.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def tokenize_texts(texts, tokenizer, max_length=256):
    """
    Tokenize all node texts once before training.

    Returns:
    - input_ids:      [N, max_length] LongTensor
    - attention_mask: [N, max_length] LongTensor
    """
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return encoded["input_ids"], encoded["attention_mask"]


def create_data_splits(y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    Create boolean masks for train / validation / test nodes.

    Inputs:
    - y: label tensor of shape [num_nodes]
    - train_ratio: fraction of nodes for training
    - val_ratio: fraction of nodes for validation
    - test_ratio: fraction of nodes for testing
    - seed: random seed

    Returns:
    - train_mask, val_mask, test_mask
    """
    num_nodes = len(y)

    torch.manual_seed(seed)
    perm = torch.randperm(num_nodes)

    train_end = int(train_ratio * num_nodes)
    val_end = train_end + int(val_ratio * num_nodes)

    train_idx = perm[:train_end]
    val_idx = perm[train_end:val_end]
    test_idx = perm[val_end:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def build_model(model_name, in_channels, hidden_channels, out_channels):
    """
    Build one GNN model by name.
    """
    if model_name == "GCN":
        model = GCN(
            in_channels=in_channels,
            all_hidden_channels=hidden_channels,
            out_channels=out_channels,
            activation=None
        )
    elif model_name == "SAGE":
        model = SAGE(
            in_channels=in_channels,
            all_hidden_channels=hidden_channels,
            out_channels=out_channels,
            activation=None
        )
    elif model_name == "GIN":
        model = GIN(
            in_channels=in_channels,
            all_hidden_channels=hidden_channels,
            out_channels=out_channels,
            activation=None
        )
    elif model_name == "GAT":
        model = GAT(
            in_channels=in_channels,
            all_hidden_channels=hidden_channels,
            out_channels=out_channels,
            activation=None
        )
    else:
        raise ValueError("Invalid model_name. Choose from 'GCN', 'SAGE', 'GIN', 'GAT'.")

    return model


# ---------------------------------------------------------------------------
# Joint model
# ---------------------------------------------------------------------------

class JointLLMGNN(torch.nn.Module):
    """
    Joint model: HuggingFace base LLM (e.g. GPT-2) + GNN.

    During the forward pass, node embeddings are computed dynamically
    from raw token tensors, so gradients flow through both the LLM
    and the GNN.

    The LLM is processed in sub-batches (llm_batch_size) to manage
    GPU memory, while the GNN always sees all nodes at once (full-graph
    message passing).
    """

    def __init__(self, llm_model_name, gnn_model_name, all_hidden_channels, out_channels):
        super(JointLLMGNN, self).__init__()

        self.llm = AutoModel.from_pretrained(llm_model_name)
        embedding_dim = self.llm.config.hidden_size  # 768 for GPT-2 base

        self.gnn = build_model(
            model_name=gnn_model_name,
            in_channels=embedding_dim,
            hidden_channels=all_hidden_channels,
            out_channels=out_channels
        )

        self.gnn_model_name = gnn_model_name

    def forward(self, input_ids, attention_mask, edge_index, edge_weight=None, llm_batch_size=8):
        """
        Forward pass.

        Inputs:
        - input_ids:      [N, seq_len]
        - attention_mask: [N, seq_len]
        - edge_index:     [2, num_edges]
        - edge_weight:    [num_edges] or None
        - llm_batch_size: nodes per LLM sub-batch

        Returns:
        - logits: [N, out_channels]
        """
        # --- LLM: compute embeddings in sub-batches ---
        # torch.cat is differentiable, so gradients flow back through
        # every sub-batch into the LLM weights.
        all_embeddings = []
        N = input_ids.shape[0]

        for i in range(0, N, llm_batch_size):
            batch_ids  = input_ids[i:i + llm_batch_size]
            batch_mask = attention_mask[i:i + llm_batch_size]
            out = self.llm(input_ids=batch_ids, attention_mask=batch_mask)
            emb = mean_pooling(out.last_hidden_state, batch_mask)
            all_embeddings.append(emb)

        X = torch.cat(all_embeddings, dim=0)  # [N, embedding_dim]

        # --- GNN: full-graph forward pass ---
        if self.gnn_model_name == "GCN":
            logits = self.gnn(X, edge_index, edge_weight)
        else:
            logits = self.gnn(X, edge_index)

        return logits


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, input_ids, attention_mask, edge_index, edge_weight, y, mask, llm_batch_size=8):
    """
    Compute loss and accuracy on the nodes selected by mask.
    """
    model.eval()

    with torch.no_grad():
        out = model(input_ids, attention_mask, edge_index, edge_weight, llm_batch_size)

        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(out[mask], y[mask])

        pred = out.argmax(dim=1)
        correct = (pred[mask] == y[mask]).sum().item()
        total = mask.sum().item()
        acc = correct / total if total > 0 else 0.0

    return loss.item(), acc


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_llm_gnn(
    model_name,
    input_ids,
    attention_mask,
    edge_index,
    edge_weight,
    y,
    train_mask,
    val_mask,
    test_mask,
    llm_model_name="gpt2",
    hidden_channels=[128],
    learning_rate=1e-4,
    weight_decay=5e-4,
    num_epochs=100,
    llm_batch_size=8
):
    """
    Train one JointLLMGNN model and report train/val/test results.
    """
    out_channels = len(torch.unique(y))

    model = JointLLMGNN(
        llm_model_name=llm_model_name,
        gnn_model_name=model_name,
        all_hidden_channels=hidden_channels,
        out_channels=out_channels
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state_dict = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        out = model(input_ids, attention_mask, edge_index, edge_weight, llm_batch_size)

        loss = loss_fn(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        train_loss, train_acc = evaluate(
            model, input_ids, attention_mask, edge_index, edge_weight,
            y, train_mask, llm_batch_size
        )
        val_loss, val_acc = evaluate(
            model, input_ids, attention_mask, edge_index, edge_weight,
            y, val_mask, llm_batch_size
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {
                key: value.clone()
                for key, value in model.state_dict().items()
            }

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_loss, test_acc = evaluate(
        model, input_ids, attention_mask, edge_index, edge_weight,
        y, test_mask, llm_batch_size
    )

    print("\nBest validation accuracy:", round(best_val_acc, 4))
    print("Test loss:", round(test_loss, 4))
    print("Test accuracy:", round(test_acc, 4))

    return model


if __name__ == "__main__":

    # -----------------------------------
    # 1. Choose model here
    # -----------------------------------
    model_name     = "GCN"
    llm_model_name = "gpt2"

    # -----------------------------------
    # 2. File paths
    # -----------------------------------
    input_folder         = "data/test_plays"
    chunked_dataset_file = "data/chunked_plays.csv"
    nodes_csv            = "data/graph_nodes.csv"
    edges_csv            = "data/graph_edges.csv"

    # -----------------------------------
    # 3. Function words
    # -----------------------------------
    function_words = [
        "the", "a", "an", "and", "or", "but", "to", "of", "in", "on",
        "for", "with", "as", "at", "by", "from", "that", "this", "it",
        "he", "she", "i", "you", "we", "they", "is", "was", "be", "been",
        "are", "were", "not", "do", "does", "did", "have", "has", "had"
    ]

    # -----------------------------------
    # 4. Build graph
    # -----------------------------------

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

    print("Nodes:", len(texts))
    print("Edges:", edge_index.shape[1])
    print("Classes:", len(torch.unique(y)))

    # -----------------------------------
    # 5. Tokenize all texts once
    # -----------------------------------
    print("\n[2] Tokenizing texts...")
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    input_ids, attention_mask = tokenize_texts(texts, tokenizer, max_length=256)

    print("input_ids shape:", input_ids.shape)

    # -----------------------------------
    # 6. Create train / val / test masks
    # -----------------------------------
    train_mask, val_mask, test_mask = create_data_splits(
        y,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42
    )

    print("\nSplit sizes:")
    print("Train nodes:", train_mask.sum().item())
    print("Val nodes:", val_mask.sum().item())
    print("Test nodes:", test_mask.sum().item())

    # -----------------------------------
    # 7. Train model
    # -----------------------------------
    model = train_llm_gnn(
        model_name=model_name,
        input_ids=input_ids,
        attention_mask=attention_mask,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        llm_model_name=llm_model_name,
        hidden_channels=[128],
        learning_rate=1e-4,
        weight_decay=5e-4,
        num_epochs=100,
        llm_batch_size=8    # reduce to 4 or 2 if GPU runs out of memory
    )