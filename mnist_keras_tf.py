import argparse, os
import numpy as np

import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import Callback, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model

if __name__ == '__main__':
    # ======================================
    # (1) Parameters for the training
    # ======================================
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--dense-layer', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.2)

    # These are setup by Sagemaker environment (or by you manually)
    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    
    args, _ = parser.parse_known_args()
    
    epochs     = args.epochs
    lr         = args.learning_rate
    batch_size = args.batch_size
    dense_layer = args.dense_layer
    dropout    = args.dropout
    
    gpu_count  = args.gpu_count
    model_dir  = args.model_dir
    training_dir   = args.training
    validation_dir = args.validation
    
    # ==========================================================
    # (2) Data from environment (either local or Sagemaker)
    # ==========================================================
    x_train = np.load(os.path.join(training_dir, 'training.npz'))['image']
    y_train = np.load(os.path.join(training_dir, 'training.npz'))['label']
    x_val  = np.load(os.path.join(validation_dir, 'validation.npz'))['image']
    y_val  = np.load(os.path.join(validation_dir, 'validation.npz'))['label']
    

    # ==========================================================
    # (3) Prepare the data for TF
    # ==========================================================

    # input image dimensions
    img_rows, img_cols = 28, 28

    # Tensorflow needs image channels last, e.g. (batch size, width, height, channels)
    # channels = 1, as we have grayscale images (for RGB images, the channels would be 3)
    if K.image_data_format() == 'channels_last':
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
        batch_norm_axis=-1
    else:
        # Keras is configured with channels first (Apache MXNet backend)
        print('Channels first, exiting')
        exit(-1)
        
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'test samples')
    
    # ===========================================
    # Normalize pixel values
    x_train  = x_train.astype('float32')
    x_val    = x_val.astype('float32')
    x_train /= 255
    x_val   /= 255
    
    # ===========================================
    # Convert class vectors to binary class matrices
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val   = keras.utils.to_categorical(y_val, num_classes)
    
    
    # ==========================================================
    # (4) NN Architecture
    # ==========================================================
    # Keras model definition
    model = Sequential()
    
    # 1st convolution block
    model.add(Conv2D(32, kernel_size=(5,5), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2))
    
    # 2nd convolution block
    #model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='valid'))
    #model.add(BatchNormalization(axis=batch_norm_axis))
    #model.add(MaxPooling2D(pool_size=(2,2), strides=2))

    # Fully connected block
    model.add(Flatten())
    model.add(Dense(dense_layer, activation='relu'))

    # Dropout layer to prevent overfitting
    model.add(Dropout(dropout))

    # Output layer (the classification layer)
    model.add(Dense(num_classes, activation='softmax'))
    
    print(model.summary())

    if gpu_count > 1:
        model = multi_gpu_model(model, gpus=gpu_count)
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
    
    # ==========================================================
    # (5) Training
    # ==========================================================
    model.fit(x_train, y_train, batch_size=batch_size,
                    validation_data=(x_val, y_val), 
                    epochs=epochs,
                    verbose=1)
    
    score = model.evaluate(x_val, y_val, verbose=0)
    print('Validation loss    :', score[0])
    print('Validation accuracy:', score[1])
    
    # ==========================================================
    # (6) Save the resulting network to file in ENV
    # ==========================================================
    sess = K.get_session()
    tf.saved_model.simple_save(
        sess,
        os.path.join(model_dir, 'model/1'),
        inputs={'inputs': model.input},
        outputs={t.name: t for t in model.outputs})
    