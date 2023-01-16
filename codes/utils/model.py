import tensorflow._api.v2.compat.v1 as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers.core import Activation
from keras.layers.normalization.batch_normalization import BatchNormalization

tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()


class Net:
    def model_contruc(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(filters=16,  # 卷积核数量
                                      kernel_size=3,  # 卷积核尺寸
                                      padding='same',  # padding补齐，让卷积之前与之后的大小相同
                                      input_shape=(11, 11, 1)))
        # model.add(BatchNormalization(axis=-1))
        model.add(Activation("relu"))

        model.add(keras.layers.SeparableConv2D(filters=16,  # 卷积核数量
                                               kernel_size=3,  # 卷积核尺寸
                                               padding='same',  # padding补齐，让卷积之前与之后的大小相同
                                               ))  # 输入维度是1通道的28*28
        model.add(BatchNormalization(axis=-1))
        model.add(Activation("relu"))
        #
        # model.add(keras.layers.SeparableConv2D(filters=16,  # 卷积核数量
        #                                        kernel_size=2,  # 卷积核尺寸
        #                                        padding='same',  # padding补齐，让卷积之前与之后的大小相同
        #                                        # 激活函数relu
        #                                        ))  # 输入维度是1通道的28*28
        # model.add(BatchNormalization(axis=-1))
        # model.add(Activation("relu"))

        model.add(keras.layers.SeparableConv2D(filters=16,  # 卷积核数量
                                               kernel_size=3,  # 卷积核尺寸
                                               padding='same')  # padding补齐，让卷积之前与之后的大小相同
                  # 激活函数relu
                  )
        model.add(BatchNormalization(axis=-1))
        model.add(Activation("relu"))

        # model.add(keras.layers.SeparableConv2D(filters=16,  # 卷积核数量
        #                                        kernel_size=3,  # 卷积核尺寸
        #                                        padding='same',  # padding补齐，让卷积之前与之后的大小相同
        #                                        ))  # 激活函数relu
        # model.add(BatchNormalization(axis=-1))
        # model.add(Activation("relu"))

        # model.add(keras.layers.noise.GaussianNoise(stddev=0.01))
        # 最大池化层
        model.add(keras.layers.MaxPool2D(pool_size=2))
        # model.add(Dropout(0.3))

        model.add(keras.layers.SeparableConv2D(filters=32,  # 卷积核数量
                                               kernel_size=2,  # 卷积核尺寸
                                               padding='same',  # padding补齐，让卷积之前与之后的大小相同
                                               ))  # 激活函数relu
        model.add(BatchNormalization(axis=-1))
        model.add(Activation("relu"))

        # model.add(keras.layers.SeparableConv2D(filters=32,  # 卷积核数量
        #                                        kernel_size=3,  # 卷积核尺寸
        #                                        padding='same',  # padding补齐，让卷积之前与之后的大小相同
        #                                        ))  # 激活函数relu
        # model.add(BatchNormalization(axis=-1))
        # model.add(Activation("relu"))

        model.add(keras.layers.SeparableConv2D(filters=32,  # 卷积核数量
                                               kernel_size=3,  # 卷积核尺寸
                                               padding='same',  # padding补齐，让卷积之前与之后的大小相同
                                               ))  # 激活函数relu
        model.add(BatchNormalization(axis=-1))
        model.add(Activation("relu"))

        # 第四层卷积层
        # model.add(keras.layers.SeparableConv2D(filters=32,  # 卷积核数量
        #                                        kernel_size=3,  # 卷积核尺寸
        #                                        padding='same',  # padding补齐，让卷积之前与之后的大小相同
        #                                        ))  # 激活函数relu
        # model.add(BatchNormalization(axis=-1))
        # model.add(Activation("relu"))

        # model.add(keras.layers.SeparableConv2D(filters=32,  # 卷积核数量
        #                                        kernel_size=3,  # 卷积核尺寸
        #                                        padding='same',  # padding补齐，让卷积之前与之后的大小相同
        #                                        ))  # 激活函数relu
        # model.add(BatchNormalization(axis=-1))
        # model.add(Activation("relu"))

        # model.add(keras.layers.SeparableConv2D(filters=64,  # 卷积核数量
        #                                        kernel_size=3,  # 卷积核尺寸
        #                                        padding='same',  # padding补齐，让卷积之前与之后的大小相同
        #                                        ))  # 激活函数relu
        # model.add(BatchNormalization(axis=-1))
        # model.add(Activation("relu"))
        #
        # model.add(keras.layers.SeparableConv2D(filters=64,  # 卷积核数量
        #                                        kernel_size=3,  # 卷积核尺寸
        #                                        padding='same',  # padding补齐，让卷积之前与之后的大小相同
        #                                        ))  # 激活函数relu
        # model.add(BatchNormalization(axis=-1))
        # model.add(Activation("relu"))

        # model.add(keras.layers.SeparableConv2D(filters=64,  # 卷积核数量
        #                                        kernel_size=3,  # 卷积核尺寸
        #                                        padding='same',  # padding补齐，让卷积之前与之后的大小相同
        #                                        ))  # 激活函数relu
        # model.add(BatchNormalization(axis=-1))
        # model.add(Activation("relu"))
        # #
        # model.add(keras.layers.SeparableConv2D(filters=64,  # 卷积核数量
        #                                        kernel_size=3,  # 卷积核尺寸
        #                                        padding='same',  # padding补齐，让卷积之前与之后的大小相同
        #                                        ))  # 激活函数relu
        # model.add(BatchNormalization(axis=-1))
        # model.add(Activation("relu"))

        model.add(keras.layers.SeparableConv2D(filters=64,  # 卷积核数量
                                               kernel_size=3,  # 卷积核尺寸
                                               padding='same',  # padding补齐，让卷积之前与之后的大小相同
                                               ))  # 激活函数relu
        model.add(BatchNormalization(axis=-1))
        model.add(Activation("relu"))
        #
        # model.add(keras.layers.SeparableConv2D(filters=128,  # 卷积核数量
        #                                        kernel_size=2,  # 卷积核尺寸
        #                                        padding='same',  # padding补齐，让卷积之前与之后的大小相同
        #                                        ))  # 激活函数relu
        # model.add(BatchNormalization(axis=-1))
        # model.add(Activation("relu"))
        # 最大池化层
        model.add(keras.layers.MaxPool2D(pool_size=2))
        model.add(Dropout(0.3))

        # 全连接层
        model.add(keras.layers.Flatten())  # 展平输出
        model.add(Dropout(0.3))
        model.add(BatchNormalization(axis=-1))
        # model.add(keras.layers.Dense(128, activation='relu'))
        # model.add(regularizers.l2(0.01))
        # model.add(keras.layers.Dense(64, kernel_regularizer=regularizers.l2(0.0001)))
        model.add(keras.layers.Dense(64))
        model.add(Dropout(0.3))
        model.add(Activation("relu"))

        model.add(keras.layers.Dense(2, activation="softmax"))

        model.compile(loss="categorical_crossentropy",
                      optimizer=keras.optimizers.SGD(0.0005),
                      metrics=["accuracy"])

        # model.summary()
        return model
