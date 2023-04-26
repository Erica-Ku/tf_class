# 미니배치 경사하강법 적용
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 데이터 생성
x_data = np.random.rand(100, 1)
y_data = 2 * x_data + 1 + 0.2 * np.random.randn(100, 1)

# 모델 아키텍처 정의
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# 손실 함수 정의
loss_fn = tf.keras.losses.MeanSquaredError()

# 옵티마이저 정의
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

# 학습할 배치 크기 설정
batch_size = 10

# 학습 데이터를 미니배치로 분할하여 학습
for epoch in range(100):
    # 미니배치로 분할한 데이터셋 생성
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).shuffle(100).batch(batch_size)
    for x_batch, y_batch in dataset:
        with tf.GradientTape() as tape:
            y_pred = model(x_batch, training=True)
            loss = loss_fn(y_batch, y_pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if (epoch+1) % 10 == 0:
        print('Epoch[{}/{}], loss: {:.4f}'.format(epoch+1, 100, loss.numpy()))

# 학습 결과 확인
x_test = np.array([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9], [1.0]])
y_test = 2 * x_test + 1
y_pred = model.predict(x_test)
print(y_pred)
plt.plot(x_test, y_test, 'o')
plt.plot(x_test, y_pred, 'r-', linewidth=3)
plt.legend(['True', 'Predicted'], loc='upper left')
plt.show()