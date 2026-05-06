# Federated Learning Contract Risk Analyser

[![GitHub](https://img.shields.io/badge/GitHub-emeringrace%2FFederated_Learning_Contract_Risk_Analyser-blue)](https://github.com/emeringrace/Federated_Learning_Contract_Risk_Analyser)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A privacy-preserving contract clause classification system built using **Federated Learning** (Flower framework) and **transformer-based NLP models** (HuggingFace). Multiple clients collaboratively train a shared global model without sharing their raw data.

**Repository:** https://github.com/emeringrace/Federated_Learning_Contract_Risk_Analyser

---

## How It Works

```
Client 1 (data/client1.csv) ─┐
Client 2 (data/client2.csv) ─┼──► Flower Server (FedAvg) ──► Global Model
Client 3 (data/client3.csv) ─┘
```

1. Each client trains locally on its own contract clause data
2. Only model weights (not raw data) are sent to the server
3. The server aggregates weights using **Federated Averaging (FedAvg)**
4. The updated global model is sent back to all clients
5. This repeats for **3 rounds**

---

## Project Structure

```
Federated_Learning_Contract_Risk_Analyser/
│
├── server.py              # Flower server with FedAvg strategy
├── client1.py             # Flower client 1
├── client2.py             # Flower client 2
├── client3.py             # Flower client 3
├── labels.py              # Label mappings and NUM_LABELS
├── split_data.py          # Script to split data across clients
│
├── data/
│   ├── client1.csv        # Training data for client 1
│   ├── client2.csv        # Training data for client 2
│   └── client3.csv        # Training data for client 3
│
├── base_model/            # Pretrained HuggingFace model (base)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── training_args.bin
│
├── global_model/          # Aggregated global model (saved after each round)
│   ├── config.json
│   └── model.safetensors
│
└── clf_df.csv             # Original dataset
```

---

## Requirements

Install dependencies:

```bash
pip install flwr torch transformers datasets scikit-learn pandas
```

---

## How to Run

Open **4 terminals** and run each command in order:

**Terminal 1 — Start the Server:**
```bash
python server.py
```

**Terminal 2 — Start Client 1:**
```bash
python client1.py
```

**Terminal 3 — Start Client 2:**
```bash
python client2.py
```

**Terminal 4 — Start Client 3:**
```bash
python client3.py
```

> ⚠️ Start the server first, then all 3 clients. The server waits for all 3 clients before beginning training.

---

## NLP: Contract Clause Classification

### Overview
The system uses **HuggingFace Transformers** for multi-class classification of contract clauses into 32 distinct risk categories.

### Supported Contract Clause Categories (32 Labels)

| Label | Description |
|-------|-------------|
| Affiliate License-Licensee | Licensing terms for affiliates as licensee |
| Affiliate License-Licensor | Licensing terms for affiliates as licensor |
| Anti-Assignment | Restrictions on assignment of contract |
| Audit Rights | Rights to audit parties' records |
| Cap On Liability | Limitation on damages or liability |
| Change Of Control | Provisions for control changes |
| Competitive Restriction Exception | Exceptions to non-compete clauses |
| Covenant Not To Sue | Agreement not to pursue legal action |
| Exclusivity | Exclusive rights or obligations |
| Expiration Date | Contract expiration terms |
| Governing Law | Jurisdiction and applicable law |
| IP Ownership Assignment | Intellectual property ownership transfer |
| Irrevocable Or Perpetual License | Long-term or permanent license grants |
| Joint IP Ownership | Shared IP ownership |
| License Grant | Permission to use intellectual property |
| Liquidated Damages | Pre-agreed damage amounts |
| Minimum Commitment | Minimum usage or purchase requirements |
| Most Favored Nation | Equal treatment clause |
| No-Solicit Of Customers | Restriction on customer solicitation |
| No-Solicit Of Employees | Restriction on employee recruitment |
| Non-Competition | Non-compete clause |
| Non-Disparagement | Non-disparagement clause |
| Non-Transferable License | Non-transferable IP rights |
| Post-Termination Services | Services after contract ends |
| Price Restrictions | Pricing control clauses |
| Renewal Term | Contract renewal conditions |
| Revenue/Profit Sharing | Revenue or profit sharing terms |
| ROFR/ROFO/ROFN | Right of first refusal/offer/negotiation |
| Source Code Escrow | Protection of source code in escrow |
| Termination For Cause | Grounds for termination |
| Termination For Convenience | Termination without cause |
| Warranty | Warranty provisions |

### NLP Architecture

**Model:** HuggingFace `AutoModelForSequenceClassification`
- **Task:** Multi-class contract clause classification
- **Input:** Contract clause text (string)
- **Output:** Predicted clause label (one of 32 categories)
- **Tokenizer:** Pretrained tokenizer from base model
- **Max Token Length:** 128 tokens
- **Framework:** PyTorch + HuggingFace Transformers

### Tokenization Pipeline

```python
# Input: Raw contract clause text
"This agreement grants an exclusive license to use the software"

# Tokenization (in client code)
tokenizer = AutoTokenizer.from_pretrained("./base_model")
tokens = tokenizer(
    text,
    truncation=True,         # Handle long clauses
    padding="max_length",    # Pad to 128 tokens
    max_length=128
)

# Output: Token IDs, attention masks
# Input to model: [input_ids, attention_mask]
# Model Output: [logits for 32 classes]
```

### Model Training Flow (per Client)

```
Raw Data (CSV)
    ↓
Load & Filter by Valid Labels
    ↓
Convert to HuggingFace Dataset
    ↓
Tokenize (truncate to 128 tokens, pad)
    ↓
Initialize Model (32 output classes)
    ↓
Fine-tune with Trainer (20 steps per round)
    ↓
Send Weights to Server (not raw data!)
    ↓
Receive Updated Weights from Server
```

### Key NLP Components in Code

| File | NLP Responsibility |
|------|-------------------|
| `labels.py` | Defines 32 contract clause labels and label-to-ID mapping |
| `client1.py`, `client2.py`, `client3.py` | Loads tokenizer, prepares datasets, trains transformer model locally |
| `base_model/` | Pretrained transformer weights and tokenizer |
| `global_model/` | Aggregated model after federated rounds |

---

## Federated Learning (FL): Architecture & Process

### What is Federated Learning?

Federated Learning is a **distributed machine learning approach** where:
1. ✅ Data stays on local clients (never sent to server)
2. ✅ Only **model weights** are shared for aggregation
3. ✅ Privacy is preserved through distributed training
4. ✅ A global model is collaboratively trained without centralized data

### Why FL for Contract Analysis?

Contract clauses contain sensitive business and legal information:
- 🔒 Companies can't share raw contracts with each other or central servers
- 🤝 Multiple parties need to collaboratively train a shared model
- 📊 Federated Learning enables this without exposing raw data

### FL Architecture: Flower Framework

**Server-Client Model:**
```
┌─────────────────────────────────────┐
│   Flower Server (FedAvg Strategy)   │
│  - Round Manager                    │
│  - Weight Aggregator                │
│  - Global Model Updater             │
└──────────┬──────────┬──────────┬────┘
           │          │          │
    ┌──────▼──┐ ┌───▼───┐ ┌────▼────┐
    │ Client 1 │ │Client 2│ │ Client 3 │
    │ FL Node  │ │FL Node │ │ FL Node  │
    └──────────┘ └───────┘ └──────────┘
```

### Training Workflow: Multi-Round Federated Averaging (FedAvg)

**Round 0 (Initialization):**
```
Server sends initial weights to all clients
```

**Round 1, 2, 3:**
```
┌─ Server ─────────────────────────────┐
│ Round 1                               │
│ 1. Request all clients to train      │
└──────────────────────────────────────┘
         ↓
┌─ Clients (Parallel) ─────────────────┐
│ Client 1:                             │
│  - Load local data (client1.csv)      │
│  - Download global weights            │
│  - Train 20 steps on local GPU/CPU    │
│  - Compute weight updates             │
│  - Send only updates to server        │
│                                       │
│ Client 2: (Same process)              │
│ Client 3: (Same process)              │
└──────────────────────────────────────┘
         ↓
┌─ Server ─────────────────────────────┐
│ Aggregation (FedAvg):                │
│ global_weights = avg(                │
│   client1_weights,                   │
│   client2_weights,                   │
│   client3_weights                    │
│ )                                     │
│ Save updated global model             │
└──────────────────────────────────────┘
         ↓
Back to Round 2, 3, ...
```

### FedAvg Algorithm

```
W(t+1) = (1/K) * Σ w_i(t)

where:
  W(t+1) = New global weights
  K      = Number of clients (3)
  w_i(t) = Weights from client i after local training
```

### Server Implementation (server.py)

**Custom `SaveModelStrategy` Class:**
- Extends `FedAvg` aggregation strategy
- Aggregates weights from 3 clients each round
- Tracks weight changes between rounds
- Saves updated global model to `global_model/` after each round

**Server Logic:**
```python
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, ...):
        # 1. Collect weights from all clients
        # 2. Average weights using FedAvg
        # 3. Compute weight changes
        # 4. Save to ./global_model/
        # 5. Return aggregated weights
```

### Client Implementation (client*.py)

**Flower Client Class:**
- `get_parameters()` - Returns current model weights
- `fit()` - Local training loop:
  - Load local CSV data
  - Tokenize with transformer tokenizer
  - Train NLP model for 20 steps
  - Return updated weights
- `evaluate()` - Report local accuracy

### Key FL Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Number of Rounds** | 3 | Federated training iterations |
| **Number of Clients** | 3 | Participants in federated setup |
| **Local Epochs** | 20 steps | Training iterations per client per round |
| **Batch Size** | 4 | Training batch size per client |
| **Aggregation Strategy** | FedAvg | Simple averaging of weights |
| **Server Address** | localhost:5040 | Communication endpoint |

### Communication Flow

```
Round 1:
├─ Server → Client 1: "Train with these weights"
├─ Server → Client 2: "Train with these weights"
├─ Server → Client 3: "Train with these weights"
│
├─ Client 1 → Server: "Here are my updated weights"
├─ Client 2 → Server: "Here are my updated weights"
├─ Client 3 → Server: "Here are my updated weights"
│
└─ Server: Aggregates → Round 2 begins

(Only weights travel over network, never raw data!)
```

### Privacy Guarantees

1. **Data Privacy:** Raw contract data never leaves client machines
2. **Model Privacy:** Only model weights shared (not intermediate activations)
3. **Aggregation Privacy:** Simple averaging (more sophisticated DP variants possible)

---

## Key Components

### Server (`server.py`)
- Uses a custom `SaveModelStrategy` extending `FedAvg`
- Aggregates model weights from all 3 clients each round
- Tracks global weight change between rounds
- Saves the updated global model after every round

### Clients (`client1.py`, `client2.py`, `client3.py`)
- Each client loads its own local CSV data
- Tokenizes contract clauses using the base model tokenizer
- Fine-tunes the model locally for 20 steps per round
- Sends updated weights to the server (never raw data)
- Reports local accuracy after each round

---

## Model

- **Base Model:** HuggingFace `AutoModelForSequenceClassification`
- **Task:** Multi-class contract clause classification
- **Max Token Length:** 128
- **Federated Rounds:** 3
- **Clients:** 3

---

## Why Federated Learning?

Contract data is sensitive — companies can't share legal documents with each other or a central server. Federated Learning allows multiple parties to collaboratively train a shared model while keeping their data **completely private and local**.

---

## Results

After 3 federated rounds, the global model is saved in the `global_model/` directory and can be used for inference on new contract clauses.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/emeringrace/Federated_Learning_Contract_Risk_Analyser.git
cd Federated_Learning_Contract_Risk_Analyser
```

### 2. Create Virtual Environment (Optional but Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install flwr torch transformers datasets scikit-learn pandas
```

### 4. Verify Installation

```bash
python -c "import flwr; import torch; import transformers; print('All dependencies installed!')"
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `flwr` | >=1.0.0 | Federated Learning framework |
| `torch` | >=1.13.0 | Deep learning backend |
| `transformers` | >=4.30.0 | HuggingFace NLP models |
| `datasets` | >=2.10.0 | Dataset handling utilities |
| `scikit-learn` | >=1.2.0 | ML utilities & metrics |
| `pandas` | >=1.5.0 | Data manipulation |

---

## Troubleshooting

### Issue: "Connection refused" when starting clients
- **Cause:** Server hasn't started yet or isn't listening on port 5040
- **Solution:** Always start `server.py` first, wait 2 seconds, then start clients

### Issue: "All 3 clients must connect"
- **Cause:** Not all 3 client scripts are running
- **Solution:** Ensure you have 4 terminals open: 1 for server, 1 for each of the 3 clients

### Issue: Out of Memory (CUDA)
- **Cause:** Batch size too large for GPU
- **Solution:** Reduce batch size in client code or use CPU (`torch.device('cpu')`)

### Issue: Model files missing in `base_model/` or `global_model/`
- **Solution:** Run `python split_data.py` to regenerate model files from pretrained weights

---

## File Descriptions

| File | Description |
|------|-------------|
| `server.py` | Flower server implementing Federated Averaging (FedAvg) |
| `client*.py` | Three clients that train locally and send weights to server |
| `labels.py` | Label mappings for contract clause classification |
| `split_data.py` | Script to split `clf_df.csv` into client-specific datasets |
| `base_model/` | Initial pretrained transformer model from HuggingFace |
| `global_model/` | Aggregated model after each federated round |
| `data/` | Client-specific training datasets (CSV files) |

---

## License

This project is licensed under the MIT License - see LICENSE file for details.

---

## References

- **Flower Framework:** https://flower.ai/
- **HuggingFace Transformers:** https://huggingface.co/transformers/
- **Federated Learning Paper:** https://arxiv.org/abs/1602.05629
- **GitHub Repository:** https://github.com/emeringrace/Federated_Learning_Contract_Risk_Analyser

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## Contact & Support

For questions or issues, please:
- Open an issue on [GitHub Issues](https://github.com/emeringrace/Federated_Learning_Contract_Risk_Analyser/issues)
- Check existing issues and documentation first
- Provide error messages and reproduction steps