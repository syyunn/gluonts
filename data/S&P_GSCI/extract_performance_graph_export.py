import pandas as pd

df = pd.read_excel("./sample.xls")

as_of = None
pad = None
end = None
for index, row in df.iterrows():
    print("hi")
    if end:
        break

    if row["Unnamed: 0"] == "As of:":
        as_of = row["Unnamed: 1"]

    if row["Unnamed: 0"] == "Effective date ":
        pad = index

    if str(row["Unnamed: 0"]) == "nan" and index > 2000:
        end = index


headers = df.iloc[pad]
df_slice = pd.DataFrame(df.values[pad + 1: end], columns=df.iloc[pad])
df_slice['Effective date '] = df_slice['Effective date '].apply(lambda x: x.strftime('%Y-%m-%d'))


if __name__ == "__main__":
    pass
