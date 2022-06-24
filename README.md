# 🎓 Time-Step Ensemble in Arithmetic 🎓
- [Time-Step Ensemble](https://github.com/aaajeong/Time-Step-Ensemble) 및 연구 이어서

앞 [연구](https://github.com/aaajeong/Time-Step-Ensemble)에서 RNN의 앙상블이 성능 향상을 이끌었다는 것을 살펴볼 수 있었다.

앞 예제에서는 기계 번역으로 time-step 마다 예측한 단어의 ensemble을 활용했다.

이렇게 time-step 마다 앙상블을 사용하여 time-series 데이터 예측의 성능 향상을 기대해보았는데,

이번에는 기계 번역이 아닌 다른 예제에 대한 성능도 확인해보려 한다.

이번에 살펴볼 문제는 **사칙 연산 계산** 이다.



참고 

- [Keras 예제 - Seq2Seq로 덧셈 구현](https://github.com/keras-team/keras/blob/2.0.0/examples/addition_rnn.py)
- [『밑바닥부터 시작하는 딥러닝 2』 (한빛미디어, 2017)](https://github.com/WegraLee/deep-learning-from-scratch-2)



#### Seq2Seq 으로 덧셈 구현 (Keras Example)


- 파일 : addition_rnn.ipynb

- 데이터셋

  - dataset/addition.txt

  - 50,000개(train:45,000, validate:5,000)

  - 795+3 _798 

    706+796_1502

    8+4  _12 

- Three digits inverted + One layer LSTM (128 HN), 50k training examples = 99% train/test accuracy in 100 epochs

- 100epoch accuracy : 0.9992



#### Seq2Seq 으로 사칙연산 구현 (Keras Example)


- 파일 : arithmetic_rnn.ipynb

- 데이터셋

  - dataset/3digit_1arith.txt

  - 50,000개(train:45,000, validate:5,000)

  - 8/245 _0.03 

    31*48 _1488

    12+5  _17

- Three digits inverted + One layer LSTM (128 HN), 50k training examples

- 100 epoch : train accuracy(**0.6234**), val_accuracy(**0.5460**) 

- 200 epoch : train accuracy( **0.6833**), val_accuracy(**0.5276**)

  - Train Loss

    ![train_loss](./img/3digit_1arith(2)(train_loss).png)

  - Train Accuracy

    ![train accuracy](./img/3digit_1arith(2)(train_accuracy).png)

  - Val_loss

    ![val loss](./img/3digit_1arith(2)(val_loss).png)

  - Val_Accuracy

    ![val accuracy](./img/3digit_1arith(2)(val_accuracy).png)


#### 👉 이 결과를 RNN Ensemble 을 사용하여 성능을 향상 시켜보자.