"""

Light version (DECUSR-L) of DECUSR model with 4 Repeating Blocks.
In this light version, feature extraction block, Direct Upsampling and Feature Upsampling  layers are removed.

This code is run via run.py file.

The algorithm is trained and evaluated via run.py for all combinations of several hyper-parameters.

To run this algorithm, make sure that this module is imported in run.py with
Uncommenting the following code in run.py (line 33 in run.py) like this:
    import_module('Decusr_4RB_light')

and make the following code (line 34 in run.py) Commented Line like this:
    # import_module('Decusr_4RB')

and then issue:
    --python.exe run.py

Hyper parameters of the model are defined in the class named 'model' residing in abstract_model.py file.
The 'model' class performs all works, e.g., training, test, evaluation, plot, etc.

Please refer to abstract_model.py for further information.

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


normalize_batch = False

class My_Model(abstract_model.model):

    def __init__(self, name='Decusr_4RB_light', training_path=None, test_path=None):
        super().__init__(name, training_path, test_path)
        
    def __get_model__(self, mode='train' ):

        metrics = self.metrics if mode=='train' else self.test_metrics

        main_input = Input(shape=self.input_shape, name='main_input')
        x = self.data_augmentation(main_input)
        x = self.fn_normalization(x)

        feature_extraction = Conv2D(self.n_filters, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(x)
        feature_extraction = Conv2D(self.n_filters, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(feature_extraction)
        feature_extraction = Conv2D(self.n_filters, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(feature_extraction)
        feature_extraction = Conv2D(self.n_filters, (1, 1), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(feature_extraction)
        if self.max_pooling:
            feature_extraction = MaxPooling2D((2,2), padding='valid')(feature_extraction)

        if self.normalize_batch:
            feature_extraction = BatchNormalization()(feature_extraction)

        # REPEATING BLOCKS #

        RB1 = Conv2D(self.n_filters, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(feature_extraction)
        RB1 = Conv2D(self.n_filters, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(RB1)
        RB1 = Conv2D(self.n_filters, (1, 1), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(RB1)
        if self.max_pooling:
            RB1 = MaxPooling2D((2,2), padding='valid')(RB1)

        if self.normalize_batch:
            RB1 = BatchNormalization()(RB1)

        if self.max_pooling:
            RB2 = concatenate([MaxPooling2D((2,2), padding='valid')(feature_extraction),  RB1])
        else:
            RB2 = concatenate([feature_extraction,  RB1])

        RB2 = Conv2D(self.n_filters, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(RB2)
        RB2 = Conv2D(self.n_filters, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(RB2)
        RB2 = Conv2D(self.n_filters, (1, 1), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(RB2)
        if self.max_pooling:
            RB2 = MaxPooling2D((2,2), padding='valid')(RB2)

        if self.normalize_batch:
            RB2 = BatchNormalization()(RB2)

        if self.max_pooling:
            RB3 = concatenate([MaxPooling2D((4, 4), padding='valid')(feature_extraction),
                               MaxPooling2D((2, 2), padding='valid')(RB1),
                               RB2])
        else:
            RB3 = concatenate([feature_extraction, RB1, RB2])

        RB3 = Conv2D(self.n_filters, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(RB3)
        RB3 = Conv2D(self.n_filters, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(RB3)
        RB3 = Conv2D(self.n_filters, (1, 1), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(RB3)
        if self.max_pooling:
            RB3 = MaxPooling2D((2,2), padding='valid')(RB3)

        if self.normalize_batch:
            RB3 = BatchNormalization()(RB3)

        if self.max_pooling:
            RB4 = concatenate([MaxPooling2D((8, 8), padding='valid')(feature_extraction),
                           MaxPooling2D((4, 4), padding='valid')(RB1),
                           MaxPooling2D((2, 2), padding='valid')(RB2),
                           RB3])
        else:
           RB4 = concatenate([feature_extraction, RB1, RB2, RB3])

        RB4 = Conv2D(self.n_filters, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(RB4)
        RB4 = Conv2D(self.n_filters, (3, 3), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(RB4)
        RB4 = Conv2D(self.n_filters, (1, 1), kernel_initializer='glorot_uniform', activation=self.activation, padding='same')(RB4)
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

        return model

