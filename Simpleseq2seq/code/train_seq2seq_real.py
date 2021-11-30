# coding: utf-8
import sys
sys.path.append('./')  
# sys.path.append('Simpleseq2seq')
import numpy as np
import matplotlib.pyplot as plt
from dataset import sequence
from common import config
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq_real, to_gpu
from seq2seq import Seq2seq
from peeky_seq2seq import PeekySeq2seq
from tqdm import tqdm
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
sys.stdout = open('2-plusminus_real_test(1).txt', 'w')

# GPU에서 실행하려면 아래 주석을 해제하세요(CuPy 필요).
# ===============================================
# config.GPU = True

# 데이터셋 읽기
(x_train, t_train), (x_test, t_test) = sequence.load_data('plusminus.txt')
char_to_id, id_to_char = sequence.get_vocab()

# 입력 반전 여부 설정 =============================================
is_reverse = True  # True
if is_reverse:
    x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]
# ================================================================

# 하이퍼파라미터 설정
vocab_size = len(char_to_id)
wordvec_size = 16
hideen_size = 128
batch_size = 128
max_epoch = 200
max_grad = 5.0

# model 개수
model_num = 5
model_list = []

# 일반 혹은 엿보기(Peeky) 설정 =====================================
model = Seq2seq(vocab_size, wordvec_size, hideen_size)
model2 = Seq2seq(vocab_size, wordvec_size, hideen_size)
model3 = Seq2seq(vocab_size, wordvec_size, hideen_size)
model4 = Seq2seq(vocab_size, wordvec_size, hideen_size)
model5 = Seq2seq(vocab_size, wordvec_size, hideen_size)
# ================================================================

# 모델 추가
model_list.append(model)
model_list.append(model2)
model_list.append(model3)
model_list.append(model4)
model_list.append(model5)

optimizer = Adam()
trainer = Trainer(model, optimizer)
trainer2 = Trainer(model2, optimizer)
trainer3 = Trainer(model3, optimizer)
trainer4 = Trainer(model4, optimizer)
trainer5 = Trainer(model5, optimizer)


acc_list = []

# gpu로 데이터 읽기
if config.GPU:
    x_train, t_train = to_gpu(x_train), to_gpu(t_train)

# max_epoch = 2
for epoch in tqdm(range(max_epoch)):
    trainer.fit(x_train, t_train, max_epoch=1,
                batch_size=batch_size, max_grad=max_grad)
    trainer2.fit(x_train, t_train, max_epoch=1,
                batch_size=batch_size, max_grad=max_grad)
    trainer3.fit(x_train, t_train, max_epoch=1,
                batch_size=batch_size, max_grad=max_grad)
    trainer4.fit(x_train, t_train, max_epoch=1,
                batch_size=batch_size, max_grad=max_grad)
    trainer5.fit(x_train, t_train, max_epoch=1,
                batch_size=batch_size, max_grad=max_grad)
    
    correct_num = 0
    for i in range(len(x_test)):    #len(x_test)
        question, correct = x_test[[i]], t_test[[i]]
        verbose = i < 10

        correct_num += eval_seq2seq_real(model_list, question, correct,
                                    id_to_char, verbose, is_reverse)

    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print('검증 정확도 %.3f%%' % (acc * 100))

num = ['1', '2', '3', '4', '5']

trainer.plot_loss(num[0], '2', 'real')
trainer2.plot_loss(num[1], '2', 'real')
trainer3.plot_loss(num[2], '2', 'real')
trainer4.plot_loss(num[3], '2', 'real')
trainer5.plot_loss(num[4], '2', 'real')

# model.save_params('plusminus_real(1).pkl')
# model2.save_params('plusminus_real(2).pkl')
# model3.save_params('plusminus_real(3).pkl')
# model4.save_params('plusminus_real(4).pkl')
# model5.save_params('plusminus_real(5).pkl')

# 그래프 그리기
x = np.arange(len(acc_list))
plt.plot(x, acc_list, marker='o')
plt.title('plusminus_real')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1.0)
plt.savefig('2-plusminus_real.png')
plt.show()



