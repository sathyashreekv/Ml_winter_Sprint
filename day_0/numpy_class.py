import numpy as np

data=np.array([20000, 25000, 30000, 35000, 40000])
new_prices=data*1.1
print("Original prices:", data)
print("New prices:", new_prices)

car=np.array([20000, 25000, 30000, 35000, 40000])
prices=car[::1]
avg_price=np.mean(prices)
normalized_prices=(prices-np.min(prices))/(np.max(prices)-np.min(prices))
print("Average price:", avg_price)
print("Normalized prices:", normalized_prices)