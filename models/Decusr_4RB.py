"""
DECUSR model with 4 Repeating Blocks. This is a DeepSR file.

To run this model, e.g., training:

--python.exe -m DeepSR.DeepSR --modelfile Decusr_4RB.py --train

Please refer to the documentation of DeepSR from the following address for other command instructions and setting:
    https://github.com/htemiz/DeepSR
"""

from tensorflow.keras import metrics
from tensorflow.keras import losses, Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,   concatenate
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, Rescaling
from tensorflow.keras.layers import Conv2D, UpSampling2D, MaxPooling2D, BatchNormalization
import tensorflow.keras.backend as K
from os.path import  dirname, abspath, basename
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import abstract_model

eps = 1.1e-6

settings = \
{
'activation': 'relu',
'augment':[], # any combination of [90,180,270, 'flipud', 'fliplr', 'flipudlr' ]
'backend': 'tensorflow',
'batchsize':2,
'channels':1,
'colormode':'RGB', # 'YCbCr' or 'RGB'
'crop': 0,
'crop_test': 6,
'decay':1e-6,
'decimation': 'bicubic',
'epatience' : 201,
'epoch':50,
'inputsize':16, #
'interp_compare': 'lanczos',
'interp_up': 'bicubic',
'lrate':1e-3,
'lrpatience': 50,
'lrfactor' : 0.5,
'metrics': ["PSNR"],
'minimumlrate' : 1e-7,
'modelname':basename(__file__).split('.')[0],
'noise':'',
'normalization':['divide', '255.0'], # ['standard', "53.28741141", "40.73203139"],
'normalizeback': False,
'normalizeground':False,
'outputdir':'',
'scale':2,
'seed': 19,
'shuffle' : True,
'stride':5, #
'target_channels': 1,
'target_cmode' : 'RGB',
'testpath' : [r'D:\working\hotelbath\test'], # change accordingly for your environment
'traindir': r"D:\working\hotelbath\train", # change accordingly for your environment
'upscaleimage': False,
'valdir': r'D:\working\hotelbath\val' , # change accordingly for your environment
'weightpath':'',
'workingdir': '',
}


if settings['scale'] == 3:
    settings['inputsize']= 11
    settings['stride'] = 4

elif settings['scale'] == 4:
    settings['inputsize']= 8
    settings['stride'] = 3

elif settings['scale'] == 8:
    settings['inputsize']= 4
    settings['stride'] = 1


class My_Model(abstract_model.model):

    def __init__(self, name='Decusr_4RB', training_path=None, test_path=None):
        super().__init__(name, training_path, test_path)

    def __get_model__(self, mode='train' ):

        metrics = self.metrics if mode=='train' else self.test_metrics

        main_input = Input(shape=self.input_shape, name='main_input')

        x = self.data_augmentation(main_input)
        x = self.fn_normalization(x)

        feature_extraction = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(x)
        feature_extraction = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(feature_extraction)
        feature_extraction = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(feature_extraction)
        feature_extraction = Conv2D(16, (1, 1), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(feature_extraction)
        if self.max_pooling:
            feature_extraction = MaxPooling2D((2, 2), padding='valid')(feature_extraction)

        if self.normalize_batch:
            feature_extraction = BatchNormalization()(feature_extraction)

        upsampler_LC = UpSampling2D(settings['scale'], name='upsampler_locally_connected')(feature_extraction)
        upsampler_direct = UpSampling2D(settings['scale'])(main_input)

        # YİNELEMELİ BLOKLAR

        if self.max_pooling:
            RB1 = concatenate([
                upsampler_LC,
                MaxPooling2D((2, 2), padding='valid')(upsampler_direct)
                               ])
        else:
            RB1 = concatenate([upsampler_LC, upsampler_direct])
        # RB1 = BatchNormalization(epsilon=eps)(RB1)
        RB1 = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(RB1)
        RB1 = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(RB1)
        RB1 = Conv2D(16, (1, 1), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(RB1)

        if self.max_pooling:
            RB1 = MaxPooling2D((2, 2), padding='valid')(RB1)

        if self.normalize_batch:
            RB1 = BatchNormalization()(RB1)

        if self.max_pooling:
            RB2 = concatenate([
                MaxPooling2D((2, 2), padding='valid')(upsampler_LC),
                MaxPooling2D((4, 4), padding='valid')(upsampler_direct),
                RB1
            ])
        else:
            RB2 = concatenate([upsampler_LC, upsampler_direct, RB1])

        RB2 = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(RB2)
        RB2 = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(RB2)
        RB2 = Conv2D(16, (1, 1), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(RB2)

        if self.max_pooling:
            RB2 = MaxPooling2D((2, 2), padding='valid')(RB2)

        if self.normalize_batch:
            RB2 = BatchNormalization()(RB2)

        if self.max_pooling:
            RB3 = concatenate([
                MaxPooling2D((4, 4), padding='valid')(upsampler_LC),
                MaxPooling2D((8, 8), padding='valid')(upsampler_direct),
                MaxPooling2D((2, 2), padding='valid')(RB1),
                RB2
            ])
        else:
            RB3 = concatenate([upsampler_LC, upsampler_direct, RB1, RB2])

        RB3 = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(RB3)
        RB3 = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(RB3)
        RB3 = Conv2D(16, (1, 1), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(RB3)

        if self.max_pooling:
            RB3 = MaxPooling2D((2, 2), padding='valid')(RB3)

        if self.normalize_batch:
            RB3 = BatchNormalization()(RB3)

        if self.max_pooling:
            RB4 = concatenate([
                MaxPooling2D((8, 8), padding='valid')(upsampler_LC),
                MaxPooling2D((16, 16), padding='valid')(upsampler_direct),
                MaxPooling2D((4, 4), padding='valid')(RB1),
                MaxPooling2D((2, 2), padding='valid')(RB2),
                RB3
            ])
        else:
            RB4 = concatenate([upsampler_LC, upsampler_direct, RB1, RB2, RB3])

        RB4 = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(RB4)
        RB4 = Conv2D(16, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(RB4)
        RB4 = Conv2D(16, (1, 1), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(RB4)

        if self.normalize_batch:
            RB4 = BatchNormalization()(RB4)

        last = Conv2D(1, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(RB4)

        flt = Flatten()(last)
        output = Dense(1, activation='sigmoid')(flt)

        model = Model(main_input, outputs=output)

        model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=metrics,
        )

        model.summary()
        # model.compile(Adam(self.lrate, self.decay), loss=losses.mean_squared_error)

        return model

