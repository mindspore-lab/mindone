import matplotlib.pyplot as plt
import pandas as pd

csv_path = "outputs/vae_celeba_train/result.log"

loss = []
"""
with open(csv_path, "r") as csvfile:
    res = list(csv.DictReader(csvfile, delimiter='\t'))
    for row in res:
        loss.append(row['loss'])
"""
df = pd.read_csv(csv_path, sep="\t", usecols=["loss"])
loss = df["loss"].to_numpy()

plt.figure(figsize=(10, 6))
# plt.title("Generator and Discriminator Loss During Training")
plt.title("Total Training Loss")
plt.plot(loss, label="AE", color="orange", linewidth=1)
# plt.plot(losses_d, label="D", color='green')
# plt.xlim(-20, 220)
# plt.ylim(0, 3.5)
plt.xlabel("steps")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_curve.png")
