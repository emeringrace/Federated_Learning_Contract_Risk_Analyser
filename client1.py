import flwr as fl
import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score
from labels import label2id, NUM_LABELS

DATA_PATH = "data/client1.csv"

df = pd.read_csv(DATA_PATH)
df = df[df["clause_label"].isin(label2id.keys())]
df["label_id"] = df["clause_label"].map(label2id)
print("Dataset size:", len(df))

tokenizer = AutoTokenizer.from_pretrained("./base_model")

def tokenize(batch):
    return tokenizer(
        batch["clause_text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

dataset = Dataset.from_pandas(df[["clause_text", "label_id"]])
dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label_id", "labels")
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

model = AutoModelForSequenceClassification.from_pretrained(
    "./base_model", num_labels=NUM_LABELS
)

training_args = TrainingArguments(
    output_dir="./tmp",
    per_device_train_batch_size=4,
    max_steps=20,
    logging_steps=5,
    save_strategy="no",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

class FLClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        trainer.train()
        return self.get_parameters({}), len(dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        outputs = trainer.predict(dataset)
        preds = outputs.predictions.argmax(axis=1)
        acc = accuracy_score(dataset["labels"], preds)
        print("Client 1 accuracy:", acc)
        return float(acc), len(dataset), {"accuracy": acc}

if __name__ == "__main__":
    client = FLClient()
    fl.client.start_client(
        server_address="localhost:8080",
        client=client.to_client(),
    )