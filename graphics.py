import pandas as pd
import matplotlib.pyplot as plt

PATH = "./Results/errors/"
df = pd.read_csv(PATH + "CNN_layers_error.csv", header=None)

plt.plot(df.values)
plt.show()
