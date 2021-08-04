import pandas as pd

file_name = './MKT-KR.xlsx'
sheet_name = 'KTBF'
df = pd.read_excel(file_name, sheet_name=sheet_name)

df = df[2:]  # row-slice
df = df.loc[:, ["시가 자동 입력", "KTB10Y"]]  # column slice

# Drop NaTs
nats = df[df['시가 자동 입력'].isnull()]
df = df.drop(nats.index)

# Replace Column Names
df.columns = ['datetime', 'KTB10Y']

df.to_csv("./KTB10Y.csv", index=False)

# import pandas as pd
# import numpy as np
# from gluonts.dataset.common import ListDataset
#
# N = 17  # number of time series
# T = 30  # number of timesteps
# prediction_length = 24
# freq = "30M"
# custom_dataset = np.random.normal(size=(N, T))
#
# start = pd.Timestamp("2020-08-28", )

# start = pd.Timestamp("01-01-2019", freq=freq)  # can be different for each time series
#
#
# # train dataset: cut the last window of length "prediction_length", add "target" and "start" fields
# train_ds = ListDataset([{'target': x, 'start': start}
#                         for x in custom_dataset[:, :-prediction_length]],
#                        freq=freq)
# # test dataset: use the whole dataset, add "target" and "start" fields
# test_ds = ListDataset([{'target': x, 'start': start}
#                        for x in custom_dataset],
#                       freq=freq)

if __name__ == "__main__":
    pass
