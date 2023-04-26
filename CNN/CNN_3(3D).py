import tensorflow as tf
import numpy as np

# 입력 이미지 크기
img_size = (64, 64, 64)

# 입력 레이어
inputs = tf.keras.layers.Input(shape=img_size + (1,))

# 3D Convolution 레이어
x = tf.keras.layers.Conv3D(32, kernel_size=(3, 3, 3), activation="relu")(inputs)
x = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

x = tf.keras.layers.Conv3D(64, kernel_size=(3, 3, 3), activation="relu")(x)
x = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

x = tf.keras.layers.Conv3D(128, kernel_size=(3, 3, 3), activation="relu")(x)
x = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

x = tf.keras.layers.Flatten()(x)

# Dense 레이어
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)

outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

# 모델 구성
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# 모델 컴파일
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# 데이터 로드 - 여기서는 예시 데이터 생성
n_samples = 1000
train_images = np.random.rand(n_samples, *img_size, 1)  # (n_train_samples, *img_size) 크기의 numpy array
train_labels = np.random.randint(2, size=n_samples)  # (n_train_samples,) 크기의 numpy array (0 또는 1)

val_images = np.random.rand(n_samples // 5, *img_size, 1)  # (n_val_samples, *img_size) 크기의 numpy array
val_labels = np.random.randint(2, size=n_samples // 5)  # (n_val_samples,) 크기의 numpy array (0 또는 1)

# 모델 학습
model.fit(
    train_images,
    train_labels,
    validation_data=(val_images, val_labels),
    epochs=10,
    batch_size=32,
)