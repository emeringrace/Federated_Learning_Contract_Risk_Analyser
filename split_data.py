import pandas as pd

df = pd.read_csv("clf_df.csv")
print("Total clauses:", len(df))

total = len(df)
split = total // 3

client1 = df.iloc[0:split][["clause_text", "category"]].rename(columns={"category": "clause_label"})
client2 = df.iloc[split:2*split][["clause_text", "category"]].rename(columns={"category": "clause_label"})
client3 = df.iloc[2*split:][["clause_text", "category"]].rename(columns={"category": "clause_label"})

client1.to_csv("data/client1.csv", index=False)
client2.to_csv("data/client2.csv", index=False)
client3.to_csv("data/client3.csv", index=False)

print("Client 1:", len(client1), "clauses")
print("Client 2:", len(client2), "clauses")
print("Client 3:", len(client3), "clauses")
print("Done!")