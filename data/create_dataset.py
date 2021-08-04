"""
start:
freq:

"""

from gluonts.dataset.field_names import FieldName

import pandas as pd
from datetime import date, timedelta

# fields = [
#     f"FieldName.{k} = '{v}'"
#     for k, v in FieldName.__dict__.items()
#     if not k.startswith("_")
# ]  # START and TARGET is required field tobe filled.

freq = '1D'
starts = []  # list of starting date for several time series
jump_btw_start = 1 # 100
num_input_steps = 100  # input period to predict "prediction length"
num_pred_series = 20
start_date = date(2010, 7, 30)   # start date
end_date = date(2020, 8, 27)   # end date
input_period = (end_date - start_date).days     # as timedelta


for i in range((input_period - num_pred_series) // jump_btw_start):
    day = str(start_date + timedelta(days=jump_btw_start*i))
    starts.append(pd.Timestamp(day, freq=freq))



if __name__ == "__main__":
    pass
