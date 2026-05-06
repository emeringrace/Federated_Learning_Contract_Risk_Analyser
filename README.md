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