# LLM + GNN for Authorship Attribution

## Pipeline

The full pipeline consists of the following steps:

---

### 1. Dataset Construction
- Raw play texts are loaded from:
  - `data/raw_texts_plays/` or `data/test_plays/`
- Each play is split into fixed-length chunks
- Output:
  - `chunked_plays.csv`

---

### 2. Graph Construction (WAN-based)

Each chunk becomes a node.  
Edges are constructed based on distances between WAN representations.

Two graph construction modes are supported:

#### (A) Pairwise Graph (Dense)
- Compute WAN distance for **all pairs of chunks**
- Convert distance to similarity weight
- Produces a **fully connected graph**

#### (B) Annoy-based Graph (Sparse)
- For each chunk:
  1. Build WAN → Markov chain → vector representation  
  2. Use **Annoy (Approximate Nearest Neighbor)** to get top-k neighbors  
  3. Compute WAN distance **only for those neighbors**
- Produces a **sparse graph with ~k edges per node**

This reduces computation from:
- **O(n²)** → **O(nk)**

Outputs:
- `graph_nodes.csv`
- `graph_edges.csv`

---

### 3. Joint LLM + GNN Learning
Instead of precomputing embeddings, this pipeline performs end-to-end training:
  - Each node (text chunk) is processed as: text → tokenizer → GPT-2 → mean pooling → embedding → GNN → prediction

---

### 4. GNN Input Preparation
- Load:
	- raw texts (one per node)
	- edge_index (graph structure)
	- edge_weight
	- labels (author)
- Tokenization is performed once before training:
	- text → input_ids + attention_mask
- Output:
	- tensors ready for joint training

---

### 5. GNN Training
- Supported models:
  - GCN
  - GraphSAGE
  - GIN
  - GAT

- Default:
  - **GCN (with edge weights)**

- Training:
	- Forward pass:
	  - GPT-2 computes embeddings
	  - GNN performs message passing
	- Loss:
	  - CrossEntropy on node labels
	- Backpropagation:
	  - Updates both GPT-2 and GNN
- Task:
	- node classification (predict author)

---

## Graph Construction Modes

You can choose graph type in `GNN_INPUT.py`:

```python
graph_mode = "pairwise"   # dense graph
graph_mode = "annoy"      # sparse graph
```
---


## Project Structure

```text
project/
├── data/
│   ├── test_plays/
│   └── raw_texts_plays/
│
├── src/
│   ├── preprocess/
│   │   ├── remove_extra_spaces.py
│   │   ├── annotate_and_mask.py
│   │   ├── split_sentences_from_annotation.py
│   │   ├── preprocess_pipeline.py
│   │   └── test_preprocess.py
│   │
│   ├── WAN/
│   │   ├── function_words.py
│   │   ├── wan_matrix.py
│   │   ├── markov_normalization.py
│   │   ├── wan_distance.py
│   │   ├── WAN_pipeline.py
│   │   ├── relative_entropy/
│   │   │   ├── Bhattacharyya_Distance.py
│   │   │   ├── Hellinger_Distance.py
│   │   │   ├── Jensen_Shannon_Divergence.py
│   │   │   ├── Kullback_Leibler_Divergence.py
│   │   │   ├── Renyi_Divergence.py
│   │   │   └── Total_Variation_Distance.py
│   │   └── test_WAN.py
│   │
│   ├── build_dataset.py
│   ├── GRAPH_CONSTRUCTION_PAIRWISE.py
│   ├── GRAPH_CONSTRUCTION_ANNOY.py
│   ├── GNN_INPUT.py
│   ├── GNN_MODELS.py
│   └── TRAIN_LLM_GNN.py
│
└── README.md
```

---

## Run full pipeline + training
```bash
python src/TRAIN_LLM_GNN.py
```

This will:
	1.	Build dataset
	2.	Construct graph (WAN)
	3.	Generate embeddings
	4.	Train LLM + GNN jointly
	5.	Print accuracy