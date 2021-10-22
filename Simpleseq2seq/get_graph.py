from matplotlib import pyplot as plt
import numpy as np

x_values = np.arange(200)
f = open("Simpleseq2seq/img/esb(soft)-avg/get_avg(esb).txt", 'r')
f1 = open("Simpleseq2seq/img/esb(survival)-avg/get_avg(survival).txt", 'r')
f2 = open("Simpleseq2seq/img/esb(real)-avg/get_avg(real).txt", 'r')
f3 = open("Simpleseq2seq/img/single_avg/get_avg(single).txt", 'r')

esb_lines = f.readlines()
sur_lines = f1.readlines()
real_lines = f2.readlines()
single_lines = f3.readlines()
f.close()
f1.close()
f2.close()
f3.close()

esb_list = list(map(float, esb_lines))
sur_list = list(map(float, sur_lines))
real_list = list(map(float, real_lines))
single_list = list(map(float, single_lines))

plt.plot(x_values, esb_list)
plt.plot(x_values, sur_list)
plt.plot(x_values, real_list)
plt.plot(x_values, single_list)

plt.legend(['Soft Voting', 'Survival', 'Majority', 'Single'])

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Addition Accuracy')
plt.show()


# y_values_1 = [10, 12, 12, 10, 14, 22, 24]
# y_values_2 = [11, 14, 15, 15, 22, 21, 12]

# plt.plot(x_values, y_values_1)
# plt.plot(x_values, y_values_2)

# plt.show()