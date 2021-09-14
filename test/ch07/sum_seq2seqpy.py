# coding: utf-8
from pickle import FALSE
import sys
sys.path.append('./')  
import numpy as np
import matplotlib.pyplot as plt
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from seq2seq import Seq2seq
from peeky_seq2seq import PeekySeq2seq

# 데이터셋 읽기
(x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt')
char_to_id, id_to_char = sequence.get_vocab()

# 입력 반전 여부 설정 =============================================
is_reverse = False  # True
if is_reverse:
    x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]
# ================================================================

# 하이퍼파라미터 설정
vocab_size = len(char_to_id)
wordvec_size = 16
hideen_size = 128
batch_size = 128
max_epoch = 25
max_grad = 5.0

# # 일반 혹은 엿보기(Peeky) 설정 =====================================
model = Seq2seq(vocab_size, wordvec_size, hideen_size)
model1 = Seq2seq(vocab_size, wordvec_size, hideen_size)
# # model = PeekySeq2seq(vocab_size, wordvec_size, hideen_size)
# # ================================================================
# optimizer = Adam()
# trainer = Trainer(model, optimizer)

model.load_params('ch07/Training/ReverseSeq.pkl')
acc_list = []
correct_num = 0
# for i in range(len(x_test)):
#     question, correct = x_test[[i]], t_test[[i]]
#     print(question)
#     print(correct)
#     verbose = i < 10
#     correct_num += eval_seq2seq(model, question, correct,
#                                 id_to_char, verbose, is_reverse)

# question, correct = x_test[[0]], t_test[[0]]
question = np.array([[1, 9, 9, 2, 8, 4, 3]])
correct = np.array([[6, 11, 4, 3]])

verbose = FALSE
correct_num += eval_seq2seq(model, question, correct,
                            id_to_char, verbose, is_reverse)


acc = float(correct_num) / len(x_test)
acc_list.append(acc)
print('검증 정확도 %.3f%%' % (acc * 100))

# # 그래프 그리기
# x = np.arange(len(acc_list))
# plt.plot(x, acc_list, marker='o')
# plt.xlabel('에폭')
# plt.ylabel('정확도')
# plt.ylim(0, 1.0)
# plt.show()

# char_to_id:  {'1': 0, '6': 1, '+': 2, '7': 3, '5': 4, ' ': 5, '_': 6, '9': 7, '2': 8, '0': 9, '3': 10, '8': 11, '4': 12}
# id_to_char:  {0: '1', 1: '6', 2: '+', 3: '7', 4: '5', 5: ' ', 6: '_', 7: '9', 8: '2', 9: '0', 10: '3', 11: '8', 12: '4'}za