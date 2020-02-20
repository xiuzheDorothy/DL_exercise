from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

plt.figure(figsize=(20,12))
m = Basemap(projection='mill',
llcrnrlat = -90,
llcrnrlon = -180,
urcrnrlat = 90,
urcrnrlon = 180)

m.drawcountries(linewidth=2)# 线宽为2的线画出国家
m.drawstates(color='b')# 蓝色线条画出州
m.drawcounties(color='darkred')
plt.show()
