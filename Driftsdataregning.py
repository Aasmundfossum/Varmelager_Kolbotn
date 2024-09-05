import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_excel('Kolbotn_driftsdata.xlsx')



print(data.columns)

power_to_HP = data['TilfÃ¸rt effekt - Varmepumpe']

plt.plot(data['Tid'], data['TilfÃ¸rt effekt - Varmepumpe'])
plt.show()