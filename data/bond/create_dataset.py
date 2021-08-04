import pandas as pd
import numpy as np
from gluonts.dataset.common import ListDataset
import matplotlib.pyplot as plt
from gluonts.dataset.util import to_pandas

df = pd.read_csv("./KTB10Y.csv")
df = df.iloc[::-1].reset_index(drop=True)

N = 17  # number of time series
T = 45  # number of time steps
prediction_length = 15
freq = "96min"

# custom_dataset = np.random.normal(size=(N, T))
starts = reversed([
    "2020-08-28 09:00:00",
    "2020-07-31 09:00:00",
    "2020-06-26 09:00:00",
    "2020-05-29 09:00:00",
    "2020-03-27 09:00:00",
    "2020-02-28 09:00:00",
    "2020-01-31 09:00:00",
    "2019-10-25 09:00:00",
    "2019-09-27 09:00:00",
    "2019-08-23 09:00:00",
    "2019-07-26 09:00:00",
    "2019-06-28 09:00:00",
    "2019-05-03 09:00:00",
    "2019-03-29 09:00:00",
    "2019-02-28 09:00:00",
    "2019-01-25 09:00:00",
    "2018-11-23 09:00:00",
    "2018-10-26 09:00:00",
    "2018-09-28 09:00:00",
    "2018-08-24 09:00:00",
    "2018-07-27 09:00:00",
    "2018-06-29 09:00:00",
    "2018-05-25 09:00:00",
    "2018-04-27 09:00:00",
    "2018-03-30 09:00:00",
    "2018-02-02 09:00:00",
    "2018-01-05 09:00:00",
    "2017-11-24 09:00:00",
    "2017-10-27 09:00:00",
    "2017-09-22 09:00:00",
    "2017-08-25 09:00:00",
    "2017-07-28 09:00:00",
    "2017-06-30 09:00:00",
    "2017-06-02 09:00:00",
    "2017-05-04 09:00:00"
])

starts = [pd.Timestamp(start) for start in starts]

df["datetime"] = df["datetime"].apply(pd.Timestamp)

fractions = []

train_dataset = []
test_dataset = []
for start in starts:
    print(start)
    index = df.index[df["datetime"] == start].tolist()[0]
    start = start.replace(hour=0, minute=0, second=0)
    time_series = df[index - 45 : index + T]["KTB10Y"]  # ['KTB10Y'].values
    train_target = time_series.values[:-prediction_length]
    test_target = time_series.values
    train_data = {"target": train_target, "start": start}
    test_data = {"target": test_target, "start": start}
    train_dataset.append(train_data)
    test_dataset.append(test_data)

train_ds = ListDataset(train_dataset, freq="96min")
test_ds = ListDataset(test_dataset, freq="96min")

train_ds_iter = iter(train_ds)
test_ds_iter = iter(test_ds)
for i in range(len(train_ds)):
    train_entry = next(train_ds_iter)
    test_entry = next(test_ds_iter)

    test_series = to_pandas(test_entry)
    train_series = to_pandas(train_entry)

    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 7))

    train_series.plot(ax=ax[0])
    ax[0].grid(which="both")
    ax[0].legend(["train series"], loc="upper left")

    test_series.plot(ax=ax[1])
    ax[1].axvline(train_series.index[-1], color='r') # end of train dataset
    ax[1].grid(which="both")
    ax[1].legend(["test series", "end of train series"], loc="upper left")

    plt.show()
    break


from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.trainer import Trainer

from gluonts.dataset.repository.datasets import get_dataset
import matplotlib.pyplot as plt
from gluonts.evaluation import Evaluator
import json


# model
# estimator = SimpleFeedForwardEstimator(
#     num_hidden_dimensions=[100],
#     prediction_length=prediction_length,
#     context_length=T-prediction_length,
#     freq=freq,
#     trainer=Trainer(ctx="cpu", epochs=1000, learning_rate=1e-4, num_batches_per_epoch=1),
# )
from gluonts.model import deepar

estimator = deepar.DeepAREstimator(freq=freq,
                                   num_layers=4,
                                   num_cells=1,
                                   prediction_length=prediction_length,
                                   trainer=Trainer(ctx="cpu", epochs=500, learning_rate=1e-4, num_batches_per_epoch=35, minimum_learning_rate=0))

predictor = estimator.train(training_data=train_ds)

from gluonts.evaluation.backtest import make_evaluation_predictions

forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,  # test dataset
    predictor=predictor,  # predictor
    num_samples=1000,  # number of sample paths we want for evaluation
)  # return vals are generators


forecasts = list(forecast_it)
tss = list(ts_it)  # acronym for "test set"

ts_entry = tss[-1]


# first entry of dataset.test
dataset_test_entry = next(iter(test_ds))

# first entry of the forecast list
forecast_entry = forecasts[-1]
print(f"Number of sample paths: {forecast_entry.num_samples}")
print(f"Dimension of samples: {forecast_entry.samples.shape}")
print(f"Start date of the forecast window: {forecast_entry.start_date}")
print(f"Frequency of the time series: {forecast_entry.freq}")
print(f"Mean of the future window:\n {forecast_entry.mean}")
print(f"0.5-quantile (median) of the future window:\n {forecast_entry.quantile(0.5)}")


def plot_prob_forecasts(ts_entry, forecast_entry):
    plot_length = 0
    prediction_intervals = (0, 0)
    legend = ["observations", "median prediction"] + [
        f"{k}% prediction interval" for k in prediction_intervals
    ][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry.plot(ax=ax)  # plot the time series
    forecast_entry.plot(show_mean=True, prediction_intervals=prediction_intervals, color="g")
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.show()

# plot_prob_forecasts(ts_entry, forecast_entry)

for i in range(len(forecasts)):
    plot_prob_forecasts(tss[i], forecasts[i])

if __name__ == "__main__":
    pass
