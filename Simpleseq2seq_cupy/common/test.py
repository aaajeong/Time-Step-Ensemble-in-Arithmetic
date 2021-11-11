import numpy as np
import cupy as cp

arr = [1, 2, 3, 4, 4, 5, 6, 6]
arr2 = [10, 15, 16, 14, 17, 17, 18, 18, 19]
vals,counts = np.unique(arr, return_counts=True)
occurences = np.where(counts == counts.max())

# id-score 딕셔너리 생성
id_score = {}
for i in range(len(arr)):
    if arr[i] not in id_score:
        id_score[arr[i]] = [arr2[i]]
    else:
        id_score[arr[i]].append(arr2[i])

# 세미정답 결정
scores = []
if len(occurences[0]) != 1:
    for i in range(len(occurences[0])):
        scores.append(np.mean(np.array(id_score[vals[occurences[0][i]]])))

    scores = np.array(scores)
    result = vals[occurences[0][np.argmax(scores)]]
else:
    result = vals[occurences[0][0]]