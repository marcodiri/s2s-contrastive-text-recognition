import pandas as pd
import seaborn as sn


metrics = pd.read_csv(f"saved_models/metrics.csv")
del metrics["step"]
del metrics["lr-Adadelta"]
del metrics["train_elapsed_time"]
del metrics["train_loss"]
del metrics["val_loss"]
metrics.set_index("epoch", inplace=True)
# display(metrics.dropna(axis=1, how="all").head())
plot = sn.relplot(data=metrics, kind="line", legend=False)
fig = plot.fig
fig.savefig("out.png") 
print()
