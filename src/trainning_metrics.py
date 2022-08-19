from dvclive import Live
import train
import evaluate
live = Live("trainning_metrics")

for epoch in range(2):
    train("./data/features", "./model.pkl")
    metrics = evaluate("./model.pkl", "./data/features")

    for metric_name, value in metrics.items():
        live.log(metric_name, value)

    live.next_step()