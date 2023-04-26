import tensorflow as tf                     # tensorflow 라이브러리를 import 한다.
from tensorflow.keras import layers, models   # tensorflow.keras에서 layers와 models 모듈을 import 한다.
from tensorflow.keras.datasets import mnist  # tensorflow.keras.datasets에서 mnist를 import 한다.

# MNIST 데이터셋 로딩
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()  # MNIST 데이터셋을 불러온다.

# 데이터 전처리 및 정규화
train_images = train_images.reshape((60000, 28, 28, 1))  # 4차원으로 reshape 한다. (데이터 개수, 이미지 너비, 이미지 높이, 채널 수)
train_images = train_images.astype('float32') / 255     # 데이터를 정규화한다. (0~1 사이 값으로 조정)
test_images = test_images.reshape((10000, 28, 28, 1))   # 4차원으로 reshape 한다. (데이터 개수, 이미지 너비, 이미지 높이, 채널 수)
test_images = test_images.astype('float32') / 255       # 데이터를 정규화한다. (0~1 사이 값으로 조정)

# 모델 구성
model = models.Sequential()                            # Sequential 모델 생성
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))  # 2D convolutional 레이어를 추가
model.add(layers.MaxPooling2D((2, 2)))                  # Max pooling 레이어를 추가
model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # 2D convolutional 레이어를 추가
model.add(layers.MaxPooling2D((2, 2)))                  # Max pooling 레이어를 추가
model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # 2D convolutional 레이어를 추가
model.add(layers.Flatten())                             # 데이터 평탄화
model.add(layers.Dense(64, activation='relu'))          # fully connected 레이어를 추가
model.add(layers.Dense(10, activation='softmax'))       # 출력 레이어를 추가

# 모델 컴파일 및 학습
model.compile(optimizer='adam',                            # adam optimizer를 사용
              loss='sparse_categorical_crossentropy',      # loss function으로 sparse categorical cross-entropy를 사용
              metrics=['accuracy'])                        # 평가 metric으로 accuracy를 사용
model.fit(train_images, train_labels, epochs=5, batch_size=64)  # 모델을 학습시킨다.

# 모델 평가
test_loss, test_acc = model.evaluate(test_images, test_labels)  # 모델을 테스트셋으로 평가
print('Test accuracy:', test_acc)  # 테스트셋 accuracy 출력
