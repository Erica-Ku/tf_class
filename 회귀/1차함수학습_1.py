import tensorflow as tf

# 1. 데이터 전처리
x_train = tf.constant([1, 2, 3, 4, 5])
y_train = tf.constant([3, 5, 7, 9, 11])

# 2. 모델 아키텍처 정의
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# 3. 손실 함수 정의
loss_fn = tf.keras.losses.MeanSquaredError()

# 4. 옵티마이저 정의
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# 5. 모델 학습
for epoch in range(1000):
    with tf.GradientTape() as tape:
        y_pred = model(x_train, training=True)
        loss = loss_fn(y_train, y_pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# 6. 모델 평가
x_test = [6, 7, 8, 9, 10]
y_test = [13, 15, 17, 19, 21]
y_pred = model.predict(x_test)
print("예측값:", y_pred)