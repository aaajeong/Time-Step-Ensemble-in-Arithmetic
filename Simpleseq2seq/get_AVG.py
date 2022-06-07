import numpy as np
import matplotlib.pyplot as plt


acc_list = []

f = open("./training_memo/PlusMinus/esb(real)_avg/1-plusminus_real_test(1).txt", 'r')
while True:
    line = f.readline()
    if line[:6] == '검증 정확도':
        acc_list.append(line[7:12])
    if not line: break
acc_list1 = list(map(float, acc_list))
f.close()

acc_list = []

f = open("./training_memo/PlusMinus/esb(real)_avg/2-plusminus_real_test(1).txt", 'r')
while True:
    line = f.readline()
    if line[:6] == '검증 정확도':
        acc_list.append(line[7:12])
    if not line: break
acc_list2 = list(map(float, acc_list))
f.close()

acc_list = []

f = open("./training_memo/PlusMinus/esb(real)_avg/3-plusminus_real_test(1).txt", 'r')
while True:
    line = f.readline()
    if line[:6] == '검증 정확도':
        acc_list.append(line[7:12])
    if not line: break
acc_list3 = list(map(float, acc_list))
f.close()

acc_list = []

f = open("./training_memo/PlusMinus/esb(real)_avg/4-plusminus_real_test(1).txt", 'r')
while True:
    line = f.readline()
    if line[:6] == '검증 정확도':
        acc_list.append(line[7:12])
    if not line: break
acc_list4 = list(map(float, acc_list))
f.close()

acc_list = []

f = open("./training_memo/PlusMinus/esb(real)_avg/5-plusminus_real_test(1).txt", 'r')
while True:
    line = f.readline()
    if line[:6] == '검증 정확도':
        acc_list.append(line[7:12])
    if not line: break
acc_list5 = list(map(float, acc_list))
f.close()

# print(acc_list1)
# print(acc_list2)
# print(acc_list3)
# print(acc_list4)
# print(acc_list5)

acc_list1 = np.array(acc_list1)
acc_list2 = np.array(acc_list2)
acc_list3 = np.array(acc_list3)
acc_list4 = np.array(acc_list4)
acc_list5 = np.array(acc_list5)

A = np.array([acc_list1, acc_list2, acc_list3, acc_list4, acc_list5])
AVG = A.mean(axis = 0)
AVG = np.around(AVG, 3)
AVG = AVG.tolist()

x = np.arange(len(AVG))
plt.plot(x, AVG, marker='o')
plt.title('PlusMinus Real Avg Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 100)
plt.savefig('PlusMinus Real Avg Accuracy.png')
plt.show()
AVG = list(map(str, AVG))
f = open('get_avg(real).txt', 'a')
for i in AVG:
    f.write(i + '\n')
f.close()

