import numpy as np


arr = np.arange((100), dtype=np.uint8)

batch_size = 10

total = int(100/batch_size)
for i in range(total):
	train_x = arr[i*batch_size:i*batch_size+batch_size]
	print(train_x)