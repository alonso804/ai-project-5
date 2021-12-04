import pandas as pd
import os

PATH = './'
EXT = '.png'

labels = {
    "covid": 0,
    "lung": 1,
    "normal": 2,
    "pneumonia": 3,
}


def complete(df, path, name):
    df[['FILE NAME']] = df.transform(
        {'FILE NAME': lambda filename: path + filename + EXT})
    df['label'] = labels[name]


dataset = {'filename': [], 'label': []}

covid = pd.read_excel('./COVID.metadata.xlsx', usecols=['FILE NAME'])
complete(covid, './COVID/', "covid")

lung = pd.read_excel('./Lung_Opacity.metadata.xlsx', usecols=['FILE NAME'])
complete(lung, './Lung_Opacity/', "lung")

normal = pd.read_excel('./Normal.metadata.xlsx', usecols=['FILE NAME'])
complete(normal, './Normal/', "normal")

pneumonia = pd.read_excel(
    './Viral Pneumonia.metadata.xlsx', usecols=['FILE NAME'])
complete(pneumonia, './Viral Pneumonia/', "pneumonia")

df = pd.concat([covid, lung, normal, pneumonia], ignore_index=0, axis=0)
df.rename(columns={"FILE NAME": "filename", "label": "label"}, inplace=True)

df.to_csv('data.csv', index=False)

print(df.tail())
