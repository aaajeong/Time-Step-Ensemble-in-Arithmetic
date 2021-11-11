import numpy as np
arr = [1, 2, 3, 4, 4, 5, 6]
vals,counts = np.unique(arr, return_counts=True)
occurences = np.where(counts == counts.max())
print(occurences[0][0])
