
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 1. 导入所需的库
print("Importing libraries...")

# 2. 准备数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
print("Data loaded and preprocessed.")

# 3. 创建模型
model = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
print("Model created.")

# 4. 编译模型
model.compile(optimizer=Adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print("Model compiled.")

# 5. 学习率调度
def lr_schedule(epoch, lr):
    return lr * tf.math.exp(-0.1 * epoch)

# 6. 训练模型
model.fit(x_train, y_train, epochs=10,
          callbacks=[LearningRateScheduler(lr_schedule)],
          validation_data=(x_test, y_test))
print("Model trained.")

# 7. 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc}")

# 8. 将模型转换为TensorFlow Lite格式
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# 9. 应用模型优化
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# 10. 保存转换后的模型
with open('model_optimized.tflite', 'wb') as f:
    f.write(tflite_model)
print("Model converted to TensorFlow Lite format with optimizations.")
