import tensorflow as tf
import numpy as np

# 1. 데이터 준비
x_train = np.array([1, 2, 3, 4, 5])
y_train = np.array([3, 5, 7, 9, 11])

# 2. 모델 아키텍처 정의
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# 3. 모델 컴파일
model.compile(optimizer='sgd', loss='mean_squared_error')

# 4. 모델 학습
model.fit(x_train, y_train, epochs=1000, verbose=0)

# 5. 학습된 모델 테스트
x_test = [6, 7, 8, 9, 10]
y_test = [13, 15, 17, 19, 21]
y_pred = model.predict(x_test)
print("예측값:", y_pred.flatten())