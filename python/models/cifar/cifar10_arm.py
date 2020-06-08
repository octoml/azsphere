from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import os

# module definition from STM
# def @main(%data: Tensor[(1, 32, 32, 4), uint8], %mean_data: Tensor[(1, 32, 32, 4), uint8], %conv0_weight: Tensor[(5, 5, 32, 4), int8], %conv0_bias: Tensor[(32), int8], %conv1_weight: Tensor[(5, 5, 32, 32), int8], %conv1_bias: Tensor[(32), int8], %conv2_weight: Tensor[(5, 5, 64, 32), int8], %conv2_bias: Tensor[(64), int8], %dense0_weight: Tensor[(10, 1024), int8], %dense0_bias: Tensor[(10), int8]) -> Tensor[(1, 10), int8] {
#   %0 = cast(%data, dtype="int16") /* ty=Tensor[(1, 32, 32, 4), int16] */;
#   %1 = cast(%mean_data, dtype="int16") /* ty=Tensor[(1, 32, 32, 4), int16] */;
#   %2 = subtract(%0, %1) /* ty=Tensor[(1, 32, 32, 4), int16] */;
#   %3 = cast(%2, dtype="int8") /* ty=Tensor[(1, 32, 32, 4), int8] */;
#   %4 = nn.conv2d(%3, %conv0_weight, padding=[2, 2, 2, 2], channels=32, kernel_size=[5, 5], data_layout="NHWC", kernel_layout="HWOI", out_dtype="int32") /* ty=Tensor[(1, 32, 32, 32), int32] */;
#   %5 = cast(%conv0_bias, dtype="int32") /* ty=Tensor[(32), int32] */;
#   %6 = nn.bias_add(%4, %5, axis=3) /* ty=Tensor[(1, 32, 32, 32), int32] */;
#   %7 = right_shift(%6, 9 /* ty=int32 */) /* ty=Tensor[(1, 32, 32, 32), int32] */;
#   %8 = cast(%7, dtype="int8") /* ty=Tensor[(1, 32, 32, 32), int8] */;
#   %9 = nn.max_pool2d(%8, pool_size=[3, 3], strides=[2, 2], layout="NHWC", ceil_mode=True) /* ty=Tensor[(1, 16, 16, 32), int8] */;
#   %10 = nn.relu(%9) /* ty=Tensor[(1, 16, 16, 32), int8] */;
#   %11 = nn.conv2d(%10, %conv1_weight, padding=[2, 2, 2, 2], channels=32, kernel_size=[5, 5], data_layout="NHWC", kernel_layout="HWOI", out_dtype="int32") /* ty=Tensor[(1, 16, 16, 32), int32] */;
#   %12 = cast(%conv1_bias, dtype="int32") /* ty=Tensor[(32), int32] */;
#   %13 = nn.bias_add(%11, %12, axis=3) /* ty=Tensor[(1, 16, 16, 32), int32] */;
#   %14 = right_shift(%13, 9 /* ty=int32 */) /* ty=Tensor[(1, 16, 16, 32), int32] */;
#   %15 = cast(%14, dtype="int8") /* ty=Tensor[(1, 16, 16, 32), int8] */;
#   %16 = nn.relu(%15) /* ty=Tensor[(1, 16, 16, 32), int8] */;
#   %17 = nn.avg_pool2d(%16, pool_size=[3, 3], strides=[2, 2], layout="NHWC", ceil_mode=True, count_include_pad=True) /* ty=Tensor[(1, 8, 8, 32), int8] */;
#   %18 = nn.conv2d(%17, %conv2_weight, padding=[2, 2, 2, 2], channels=64, kernel_size=[5, 5], data_layout="NHWC", kernel_layout="HWOI", out_dtype="int32") /* ty=Tensor[(1, 8, 8, 64), int32] */;
#   %19 = cast(%conv2_bias, dtype="int32") /* ty=Tensor[(64), int32] */;
#   %20 = nn.bias_add(%18, %19, axis=3) /* ty=Tensor[(1, 8, 8, 64), int32] */;
#   %21 = right_shift(%20, 9 /* ty=int32 */) /* ty=Tensor[(1, 8, 8, 64), int32] */;
#   %22 = cast(%21, dtype="int8") /* ty=Tensor[(1, 8, 8, 64), int8] */;
#   %23 = nn.relu(%22) /* ty=Tensor[(1, 8, 8, 64), int8] */;
#   %24 = nn.avg_pool2d(%23, pool_size=[3, 3], strides=[2, 2], layout="NHWC", ceil_mode=True, count_include_pad=True) /* ty=Tensor[(1, 4, 4, 64), int8] */;
#   %25 = nn.batch_flatten(%24) /* ty=Tensor[(1, 1024), int8] */;
#   %26 = nn.dense(%25, %dense0_weight, units=10, out_dtype="int32") /* ty=Tensor[(1, 10), int32] */;
#   %27 = cast(%dense0_bias, dtype="int32") /* ty=Tensor[(10), int32] */;
#   %28 = left_shift(%27, 3 /* ty=int32 */) /* ty=Tensor[(10), int32] */;
#   %29 = nn.bias_add(%26, %28, axis=-1) /* ty=Tensor[(1, 10), int32] */;
#   %30 = right_shift(%29, 5 /* ty=int32 */) /* ty=Tensor[(1, 10), int32] */;
#   cast(%30, dtype="int8") /* ty=Tensor[(1, 10), int8] */
# }

def build_and_train():
    batch_size = 32
    num_classes = 10
    epochs = 1
    data_augmentation = True
    num_predictions = 20
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = f'cifar10_arm'
    input_shape = (None, 32, 32, 3)

    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential(name='cifar10_arm')
    #conv 0
    model.add(keras.layers.ZeroPadding2D(padding=(2, 2)))
    model.add(Conv2D(32, (5, 5), padding='valid',
                    input_shape=x_train.shape[1:]))

    #maxpool 0
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Activation('relu'))

    #conv 1
    model.add(keras.layers.ZeroPadding2D(padding=(2, 2)))
    model.add(Conv2D(32, (5, 5)))
    model.add(Activation('relu'))

    # average_pool 0
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    #conv 2
    model.add(keras.layers.ZeroPadding2D(padding=(2, 2)))
    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))

    # average_pool 1
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    #flatten
    model.add(Flatten())

    #dense
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.build(input_shape)
    print(model.summary())

    # initiate RMSprop optimizer
    opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    checkpoint = ModelCheckpoint(os.path.join(save_dir, f'{model_name}_best.h5'), monitor='loss', verbose=1,
        save_best_only=True, mode='auto', period=1)

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test, y_test),
                shuffle=True,
                callbacks=[checkpoint])
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train,
                                        batch_size=batch_size),
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            workers=4,
                            callbacks=[checkpoint])

    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # Score best trained model.
    best_model = load_model(os.path.join(save_dir, f'{model_name}_best.h5'))
    scores = best_model.evaluate(x_test, y_test, verbose=1)
    print('Best Test loss:', scores[0])
    print('Best Test accuracy:', scores[1])

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

def gen_custom_cifar_keras(shape_dict):
    num_classes = 10
    epochs = 1
    input_shape = (None, 32, 32, 3)

    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    model = Sequential(name='cifar10_arm')
    #conv 0
    model.add(keras.layers.ZeroPadding2D(padding=(2, 2)))
    model.add(Conv2D(8, (5, 5), padding='valid',
                    input_shape=x_train.shape[1:]))

    #maxpool 0
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Activation('relu'))

    #conv 1
    model.add(keras.layers.ZeroPadding2D(padding=(2, 2)))
    model.add(Conv2D(15, (5, 5)))
    model.add(Activation('relu'))

    # average_pool 0
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    #conv 2
    model.add(keras.layers.ZeroPadding2D(padding=(2, 2)))
    model.add(Conv2D(16, (5, 5)))
    model.add(Activation('relu'))

    # average_pool 1
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    #flatten
    model.add(Flatten())

    #dense
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.build(input_shape)
    print(model.summary())

    # initiate RMSprop optimizer
    opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

    return model

def get_cifar_keras(filepath, shape_dict):
    model = load_model(filepath)
    return model

if __name__ == '__main__':
    build_and_train()