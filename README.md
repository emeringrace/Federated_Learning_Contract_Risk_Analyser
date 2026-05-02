# Federated Learning Contract Risk Analyser

A privacy-preserving contract clause classification system built using **Federated Learning** (Flower framework) and **transformer-based NLP models** (HuggingFace). Multiple clients collaboratively train a shared global model without sharing their raw data.

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