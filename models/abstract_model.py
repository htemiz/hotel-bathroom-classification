
from os import listdir, makedirs, remove, environ
from os.path import isfile, join, exists, splitext, abspath, basename
import glob
import numpy as np
from pathlib import Path
import csv
import shutil
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, Rescaling
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.framework.config import set_visible_devices, list_physical_devices
from tensorflow.keras import layers
from tensorflow import device
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC, Accuracy, Precision, Recall, BinaryAccuracy
from tensorflow.keras.metrics import TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from tensorflow.keras.metrics import PrecisionAtRecall, SensitivityAtSpecificity, SpecificityAtSensitivity
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay, PrecisionRecallDisplay
from sklearn. model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

import metrics

from os import listdir, getcwd, makedirs, chdir, curdir
from os.path import join, basename, isdir, exists
import pickle
from time import time

class model(object):

    def set_folders(self):
        self.output_folder = join(getcwd(), 'output/' + self.name)

        if not exists(self.output_folder):
            makedirs(self.output_folder)

        self.weight_path = self.output_folder


    def __init__(self, name=None, training_path=None, test_path=None, input_shape=(512,512,3), lrate= 1e-3, gpu=1 ):
        """ takes arguments via a dictionary object and generate class members

        """
        self.name = name
        self.lrate = lrate
        self.optimizer = keras.optimizers.Adam(self.lrate)
        self.loss = 'categorical_crossentropy'
        self.metrics = ['categorical_accuracy']
        self.gpu = gpu
        self.test_metrics = [ 'categorical_accuracy', Precision(), Recall(), AUC(), TruePositives(), TrueNegatives(),
                         FalsePositives(), FalseNegatives(),
                         ] #,'accuracy', 'AUC']

        self.normalize_batch = True
        self.training_path = training_path
        self.test_path = test_path
        self.epochs = 500
        self.batch_size = 32
        self.save_best = True
        self.monitor = 'val_loss'
        self.monitoring_mode = 'auto'
        self.log_file_name= self.name + '_training.log'
        self.val_split=0.1
        self.n_filters = 8
        self.max_pooling=True
        self.figures =[]
        self.label_mode = False

        self.set_folders()
        
        self.input_shape = input_shape
        self.activation = 'relu'
        self.padding = 'same'
        self.kernel_initializer = 'glorot_uniform'
        self.seed = 19
        self.test_batch_size = 15
        self.lrpatience = 100
        self.lrfactor = 0.5
        self.min_lr = 1e-8
        self.espatience=501

        self.data_augmentation = keras.Sequential(
            [RandomFlip('horizontal'),
             RandomRotation(0.1),
             RandomRotation(0.15),
             RandomRotation(0.25),
             ]
        )
        self.exp = 1/255.
        self.fn_normalization= Rescaling(1./255.)
        
        self.verbose=2

        self.prepare_callbacks()

    def prepare_callbacks(self):

        self.callbacks = [
            ModelCheckpoint(join(self.output_folder, self.name + '_{epoch:04d}.h5'), save_best_only=self.save_best,
                                            monitor=self.monitor, mode=self.monitoring_mode),

            CSVLogger(join(self.output_folder, self.log_file_name)),

            ReduceLROnPlateau(monitor=self.monitor, factor=self.lrfactor,
                              patience=self.lrpatience, min_lr=self.min_lr, verbose=self.verbose),

            EarlyStopping(monitor=self.monitor, patience=self.espatience, verbose=self.verbose)
        ]


    def __train__(self, training_path=None, train_ds=None, val_ds=None, weight_path=None):
        
        if training_path is not None:
            self.training_path = training_path

        if train_ds is None:
            train_ds = image_dataset_from_directory(
                self.training_path,
                validation_split=self.val_split,
                seed=self.seed,
                subset="training",
                image_size=self.input_shape[0:2],
                label_mode='categorical',
                batch_size=self.batch_size,
            )
            
        if val_ds is None:
            val_ds = image_dataset_from_directory(
                self.training_path,
                validation_split=self.val_split,
                seed=self.seed,
                subset="validation",
                image_size=self.input_shape[0:2],
                label_mode='categorical',
                batch_size=self.batch_size,
            )


        # prefetch the data to yield data from disk without I/O becoming blocking
        self.train_ds = train_ds.prefetch(buffer_size=self.batch_size)
        self.val_ds = val_ds.prefetch(buffer_size=self.batch_size)

        self.model =self.__get_model__(mode='train')

        if weight_path is not None:
            self.model.load_weights(weight_path)


        start = time()
        self.history = self.model.fit(
            self.train_ds, epochs=self.epochs, callbacks=self.callbacks,
            validation_data=self.val_ds,
            verbose=self.verbose
        )
        
        elapsed_time = (time() - start) / 60. # elapsed time in minutes

        self.hist_df = pd.DataFrame(self.history.history)

        with open(join(self.output_folder, self.name + '_history.csv'), mode='w') as f:
            self.hist_df.to_csv(f)

        # np.save(join(self.output_folder, self.name + '_history.npy'), self.history.history)


        return self.history, elapsed_time

    def __get_true_labels__(self, ds=None):
        """
        Tensorflow Dataset içindeki gerçek etiket değerlerini verir.
        :param ds:
        :return: ndarray, y_true değerleri
        """
        if ds is None:
            ds = image_dataset_from_directory(
                self.test_path,
                validation_split=None,
                seed=self.seed,
                # subset="test",
                image_size=self.input_shape[0:2],
                label_mode='binary',
                batch_size=1,
            )

        # her tensorflow dataset içindeki imge ve etiketler
        # indeks, etiket şeklinde bulunur. dolayısı ile ,y etiketi verir.
        return np.concatenate([y for x, y in ds], axis=0)


    def __test__(self, test_path=None, test_ds=None, weight_path=None ):

        if test_path is not None:
            self.test_path = test_path
        
        if not exists(self.output_folder):
            makedirs(self.output_folder)

        if test_ds is None:
            test_ds = image_dataset_from_directory(
                self.test_path,
                validation_split=None,
                seed=self.seed,
                # subset="test",
                image_size=self.input_shape[0:2],
                label_mode='categorical',
                batch_size=1,#self.test_batch_size,
            )

        n_files  = test_ds.file_paths.__len__()

        # if not hasattr(self, 'model') or self.model is None:
        self.model = self.__get_model__(mode='test')

        index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=["model", "weight"])
        col_names = ['loss']
        col_names += list([x.name.rsplit('_',1)[0] if not isinstance(x, str) else x for x in self.test_metrics]) + ['average_time']

        self.df_test = pd.DataFrame(columns=col_names, index=index)

        if weight_path is not None:
            weight_files = (weight_path,)

        else:
            if isdir(self.weight_path):
                weight_files = listdir(self.weight_path)  # list current folder

                weight_files = [join(self.weight_path, x) for x in weight_files if '.h5' in x]
            else:
                weight_files = [self.weight_path]

        with device('/gpu:' + str(self.gpu)):
            for weight_file in weight_files:
                print(basename(weight_file))

                self.model.load_weights(weight_file)

                start = time()
                res = self.model.evaluate(test_ds, verbose=self.verbose)
                elapsed_time = (time() - start) / (n_files * 60.) # elapsed time in minutes
                self.df_test.loc[(self.name, basename(weight_file)), :] = [*res, elapsed_time]

        self.df_test.to_excel(join(self.output_folder, self.name + '_test_results.xlsx'))
        return self.df_test


    def __get_model__(self, mode='train'):
        raise NotImplementedError


    def plot_roc_curve(self, fpr, tpr, roc_auc, n_classes=5, save=False, titles=None, labels=None,  xlim=None, ylim=None, dpi=400):
        figures = list()
        fig = plt.figure(dpi=dpi)
        legend_texts = list()

        linestyle = "-"
        lw = 2.25

        for i in range(len(fpr)):

            if i < n_classes:
                key = i
                if labels is not None and n_classes == len(labels):
                    nam = labels[i]
                else:
                    nam=''

                label = str(nam) + " (AUC = %0.3f)" % roc_auc[key]
            else:
                key = list(fpr)[i]
                label = "All_" + key + " (AUC = %0.3f)" % roc_auc[key]

            if key in ('micro',):
                linestyle = ':'
            elif key in ('macro',):
                linestyle = '--'

            plt.plot(
                fpr[key],
                tpr[key],
                # color="darkorange",
                linestyle=linestyle,
                lw=lw,
                label=label,
            )

            plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")  #
            plt.xlim([-0.02, 1.02])
            plt.ylim([-0.02, 1.02])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            if titles is not None:
                plt.title(titles[key])
            else:
                plt.title("ROC Curve ")

        plt.legend(loc="lower right", title="Classes")

        figures.append(fig)
        if save:
            plt.savefig("ROC Curve.png")
        # plt.tight_layout()
        plt.show()

        return figures

    def __predict__(self, model=None,  test_path=None, test_ds=None, ):

        if test_path is not None:
            self.test_path = test_path

        if not exists(self.output_folder):
            makedirs(self.output_folder)

        if test_ds is None:
            test_ds = image_dataset_from_directory(
                self.test_path,
                validation_split=None,
                seed=self.seed,
                # subset="test",
                image_size=self.input_shape[0:2],
                label_mode='categorical',
                batch_size=self.test_batch_size,
            )

        if model is None and hasattr(self, 'model'):
            model = self.model
        else:
            model = self.__get_model__(mode='test')

        n_files = np.concatenate([y for x, y in test_ds], axis=0).shape[0]

        with device('/gpu:' + str(self.gpu)):
            start = time()
            res = model.predict(test_ds, verbose=self.verbose)
            elapsed_time = (time() - start) / (n_files * 60.)  # elapsed time in minutes

        return res, elapsed_time



    ## -- BURADAN SONRAKİ KODLAR ai_model.py DOSYASINDAN ALINDI -- ##

    def __roc_hesapla__(self, y_test, y_score, y_proba, plot=False, labels=None, xlim=None, ylim=None, dpi=400):

        """
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
        https://stackoverflow.com/questions/70278059/plotting-the-roc-curve-for-a-multiclass-problem
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

        :param y_test:
        :param y_score:
        :param n_classes:
        :param plot:
        :param save:
        :param titles:
        :return: fpr, tpr, roc_auc ve varsa figürler.
        """

        figures = None

        # ndim=1 ise, etiketler tek boyutlu verilmiş demektir.
        if y_test.ndim == 1 and self.n_classes > 2:
            y_test = label_binarize(y_test, classes=np.arange(self.n_classes))

        # ndim=1 ise, etiketler tek boyutlu verilmiş demektir.
        if y_score.ndim == 1 and self.n_classes > 2:
            y_score = label_binarize(y_score, classes=np.arange(self.n_classes))

        # ----------------------------------------------------------------- #
        #              MICRO AVERAGE YÖNTEMİNE GÖRE FPR, TPR VE AUC
        #
        # https://stackoverflow.com/questions/70278059/plotting-the-roc-curve-for-a-multiclass-problem
        # ----------------------------------------------------------------- #
        # fpr, tpr, pos_labels, roc_auc = metrics.roc_micro(y_test, y_score, self.n_classes)
        fpr, tpr, pos_labels, roc_auc = metrics.roc_micro(y_test, y_proba, self.n_classes)

        # ----------------------------------------------------------------- #
        #              MACRO AVERAGE YÖNTEMİNE GÖRE FPR, TPR VE AUC
        # ----------------------------------------------------------------- #
        fpr, tpr, roc_auc = metrics.roc_macro(fpr, tpr, roc_auc, self.n_classes)

        figure = metrics.plot_roc_curve(fpr, tpr, roc_auc, n_classes=self.n_classes,
                                        save=False, plot=plot, titles=None, labels=labels,
                                        xlim=xlim, ylim=ylim, dpi=dpi)

        if not hasattr(self, 'calculated_metrics'):
            self.calculated_metrics = dict()

        micro_roc_auc_ovo = roc_auc_score(y_test, y_score, multi_class="ovo", average="micro")

        macro_roc_auc_ovo = roc_auc_score(y_test, y_score, multi_class="ovo", average="macro")

        macro_roc_auc_ovr = roc_auc_score(y_test, y_score, multi_class="ovr", average="macro")

        micro_roc_auc_ovr = roc_auc_score(y_test, y_score, multi_class="ovr", average="micro")

        weighted_roc_auc_ovo = roc_auc_score(y_test, y_score, multi_class="ovo", average="weighted")

        weighted_roc_auc_ovr = roc_auc_score(y_test, y_score, multi_class="ovr", average="weighted")

        self.calculated_metrics['auc_ovo_micro'] = micro_roc_auc_ovo
        self.calculated_metrics['auc_ovo_macro'] = macro_roc_auc_ovo
        self.calculated_metrics['auc_ovr_macro'] = macro_roc_auc_ovr
        self.calculated_metrics['auc_ovr_micro'] =  micro_roc_auc_ovr
        self.calculated_metrics['auc_ovo_weighted'] = weighted_roc_auc_ovo
        self.calculated_metrics['auc_ovr_weighted'] = weighted_roc_auc_ovr
        self.figures.append(figure)

        # print(
        #     "\nOne-vs-One ROC AUC scores:\n{:.3f} (macro),\n{:.3f} "
        #     "(weighted by prevalence)".format(macro_roc_auc_ovo, weighted_roc_auc_ovo)
        # )
        # print(
        #     "\nOne-vs-Rest ROC AUC scores:\n{:.3f} (macro),\n{:.3f} "
        #     "(weighted by prevalence)".format(macro_roc_auc_ovr, weighted_roc_auc_ovr)
        # )

        return fpr, tpr, pos_labels, roc_auc, figures


    def __calculate_confusion_matrix__(self, Y_test, y_score, plot=False, labels=None, dpi=400):
        """

        :param Y_test:
        :param y_score:
        :param plot:
        :return:
        """
        if self.label_mode: # Y (etiketler) LabelEncoder ile kodlanmış ise
            self.confusion_matrix = confusion_matrix(Y_test, y_score,)
        else:
            self.confusion_matrix = confusion_matrix(np.argmax(Y_test, axis=1), y_score.argmax(axis=1))

        if plot:
            fig, ax = metrics.plot_confusion_matrix(self.confusion_matrix, on_screen=plot, plot=plot, labels=labels, dpi=dpi)
            self.figures.append(fig)

        if hasattr(self, 'calculated_metrics'):
            self.calculated_metrics['confusion_matrix'] = self.confusion_matrix

        return self.confusion_matrix, fig


    def __precision_recall_curve__(self, y_test, y_score, plot=False, labels=None, dpi=400):
        """

        https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
        https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics

        :param y_test:
        :param y_score:
        :return:
        """


        """
        DiĞER KOD
        from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
        from sklearn.metrics import average_precision_score

        precs, recs, threshes = precision_recall_curve(ytrue, ypred)
        avg_prec = average_precision_score(ytrue, ypred)
        disp = PrecisionRecallDisplay(precs, recs, average_precision=avg_prec)
        disp.plot()
        plt.show()
        return disp.figure_
        """


        # ndim=1 ise, etiketler tek boyutlu verilmiş demektir.
        if y_test.ndim == 1 and self.n_classes > 2:
            y_test = label_binarize(y_test, classes=np.arange(self.n_classes))

        # ndim=1 ise, etiketler tek boyutlu verilmiş demektir.
        if y_score.ndim == 1 and self.n_classes > 2:
            y_score = label_binarize(y_score, classes=np.arange(self.n_classes))


        precs, recs, thresholds, avg_prec = dict(), dict(), dict(), dict()

        for i in range(self.n_classes):
            precs[i], recs[i], thresholds[i] = precision_recall_curve(y_test[:,i], y_score[:,i])
            avg_prec[i] = average_precision_score(y_test[:,i], y_score[:,i])


        # micro average
        precs['micro'], recs['micro'], thresholds['micro'] = precision_recall_curve(y_test.ravel(), y_score.ravel())
        avg_prec['micro'] = average_precision_score(y_test, y_score, average='micro')

        # macro average.
        precs['macro'], recs['macro'], thresholds['macro'] = precision_recall_curve(y_test.ravel(), y_score.ravel())

        avg_prec['macro'] = average_precision_score(y_test, y_score, average='macro')

        counts = np.sum(y_test, axis=0)
        plain_thresholds, weighted_thresholds, thresholds= \
            self.__find_optimal_thresholds__(precs, recs, counts )

        #
        #   HEPSİNİ BİRDEN YAZDIRALIM
        #
        fig, ax = plt.subplots(dpi=dpi)
        for i in range(self.n_classes):
            # precision, recall, _ = precision_recall_curve(y_test[:, i], y_score[:, i])
            display = PrecisionRecallDisplay(precs[i], recs[i],
                                             average_precision=avg_prec[i] )


            if  labels is not None and self.n_classes == len(labels):
                nam = str(labels[i])
            else:
                nam = None
            display.plot( ax=ax, name=nam )

        disp = PrecisionRecallDisplay(recall=recs['micro'], precision=precs['micro'],
                                      average_precision=avg_prec['micro'])

        # plt.scatter(self.plain_threshold, self.weighted_threshold, marker='o', color='black', label='best')

        disp.plot( ax=ax, name="Average (micro)", )

        f_scores = np.linspace(0.2, 0.8, num=4)
        lines, labels = [], []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
            plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

        # add the legend for the iso-f1 curves
        handles, labels = disp.ax_.get_legend_handles_labels()
        handles.extend([l])
        labels.extend(["iso-f1 curves"])
        # set the legend and the axes
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.legend(handles=handles, labels=labels, handlelength=1, ncol=1,  fontsize='small')
        ax.set_title("Precision-Recall")

        if plot:
            plt.show()

        self.figures.append(fig)

        return precs, recs, avg_prec, thresholds, fig


    def __find_optimal_thresholds__(self, precs, recs, counts ):

        """
        Precision ve Recall değerlerine göre en optimum eşik (threshold) değerini verir.

        Threshold'u manuel aramak için:
        https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/

        :param precs:Precision values.
        :param recs: Recall values.
        :param counts: Number of total samples of each class as array.
        :return: Two Thresholds computed with simple mean, and weighted mean taking into account
        number of samples of each class.
        """

        f_scores = [(2 * precs[i] * recs[i]) / (precs[i] + recs[i]) for i in range(self.n_classes)]

        f_score_idxs = [np.argmax(f_scores[i]) for i in range(self.n_classes)]

        self.pr_f_score_idxs = f_score_idxs

        threshes = [f_scores[i][k] for i, k in enumerate(f_score_idxs)]
        threshes = np.nan_to_num(threshes)

        total = np.sum(counts)

        plain_threshold = np.mean(threshes)

        weighted_threshold = np.sum(threshes * counts) / total

        self.pr_plain_threshold = plain_threshold
        self.pr_weighted_threshold = weighted_threshold
        self.pr_thresholds = threshes

        return plain_threshold, weighted_threshold, threshes


    def __calculate_metrics__(self, y_test, y_score,):
        """

        https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics


        :param y_test:
        :param y_score:
        :return:
        """
        # ndim=1 ise, etiketler tek boyutlu verilmiş demektir.
        if y_test.ndim == 1 and self.n_classes > 2:
            y_test = label_binarize(y_test, classes=np.arange(self.n_classes))

        # ndim=1 ise, etiketler tek boyutlu verilmiş demektir.
        if y_score.ndim == 1 and self.n_classes > 2:
            y_score = label_binarize(y_score, classes=np.arange(self.n_classes))

        self.metrics = dict()
        self.metrics['accuracy'] = accuracy_score(y_test, y_score)
        self.metrics['balanced_accuracy'] = balanced_accuracy_score(y_test.ravel(), y_score.ravel())
        self.metrics['f1_micro'] = f1_score(y_test, y_score, average='micro')
        self.metrics['f1_macro'] = f1_score(y_test, y_score, average='macro')
        self.metrics['precision_micro'] = precision_score(y_test, y_score, average='micro')
        self.metrics['precision_macro'] = precision_score(y_test, y_score, average='macro')
        self.metrics['recall_micro'] = recall_score(y_test, y_score, average='micro')
        self.metrics['recall_macro'] = recall_score(y_test, y_score, average='macro', zero_division='warn')

        # Area under ROC for the multiclass problem
        self.metrics['auc_macro_ovo'] = roc_auc_score(y_test, y_score, multi_class="ovo", average="macro")
        self.metrics['auc_weighted_ovo'] = roc_auc_score(y_test, y_score, multi_class='ovo', average='weighted')
        self.metrics['auc_macro_ovr'] = roc_auc_score(y_test, y_score, multi_class="ovr", average="macro")
        self.metrics['auc_weighted_ovr'] = roc_auc_score(y_test, y_score, multi_class='ovr', average='weighted')

        self.metrics['classification_report'] =  classification_report(y_test, y_score, digits=3 )

        return self.metrics


    def __print_metrics__(self,):

        for k, v in self.calculated_metrics.items():
            if k == 'classification_report':
                print(k, '\n', v)
            else:
                print(k, v)
