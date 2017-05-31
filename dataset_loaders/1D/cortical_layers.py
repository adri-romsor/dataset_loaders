import os
import time

import numpy as np
from PIL import Image
import re
import warnings
from matplotlib import cm

from dataset_loaders.parallel_loader import ThreadedDataset
# from parallel_loader_1D import ThreadedDataset_1D

floatX = 'float32'

class CorticalLayersDataset(ThreadedDataset):
    '''The Cortical Layers dataset consists of a number of 1D intensity profiles
    extracted from the BigBrain. The goal is to perform semantic segmentation of
    the 1D intensity profiles. The dataset contains a train/validation set with
    their corresponding labels, as well as a test set without labels. The dataset
    accounts for N labels, including N-1 cortical layer labels and a padding
    label, corresponding to regions outside the cortical layers.
    Parameters
    ----------
    which_set: string
        A string in ['train', 'val', 'valid', 'test'], corresponding to
        the set to be returned.
    split: float32
        A float indicanting the fraction of data to be used for training. The
        remaining data will be used for validation.
    smooth_or_raw: string
        A string in ['raw', 'smooth', 'both'], correspoding to the channels to be
        returned. Raw are profiles sampled from the raw image, smooth profiles
        are the same profiles but anisotropically smoothed - mainly tangentially
        to the profiles.
    preload: bool
        Whether to preload all the images in memory (as a list of
        ndarrays) to minimize disk access.
    n_layers: int
        Integer indicating the number of cortical layers to segment.
    '''
    name = 'cortical_layers'

    non_void_nclasses = 7
    GTclasses = [0, 1, 2, 3, 4, 5, 6]
    _cmap = {
        0: (128, 128, 128),    # padding
        1: (128, 0, 0),        # layer 1
        2: (128, 64, ),        # layer 2
        3: (128, 64, 128),     # layer 3
        4: (0, 0, 128),        # layer 4
        5: (0, 0, 64),         # layer 5
        6: (64, 64, 128),      # layer 6
    }
    _mask_labels = {0: 'padding', 1: 'layers1', 2: 'layer2', 3: 'layer3',
                    4: 'layer4', 5: 'layer5',   6: 'layer6'}
    _void_labels = []


    _filenames = None

    @property
    def filenames(self):

        if self._filenames is None:
            # Load filenames
            nfiles = sum(1 for line in open(self.mask_path))
            filenames = range(nfiles)
            np.random.seed(1609)
            np.random.shuffle(filenames)

            if self.which_set == 'train':
                filenames = filenames[:int(nfiles*self.split)]
            elif self.which_set == 'val':
                filenames = filenames[-(nfiles - int(nfiles*self.split)):]

            # Save the filenames list
            self._filenames = filenames

        return self._filenames

    def __init__(self,
                 which_set="train",
                 split=0.85,
                 smooth_or_raw='both',
                 preload=False,
                 n_layers=6,
                 *args, **kwargs):

        n_layers_path = str(n_layers)+"layers_segmentation"

        self.which_set = "val" if which_set == "valid" else which_set
        if self.which_set not in ("train", "val", 'test'):
            raise ValueError("Unknown argument to which_set %s" %
                             self.which_set)

        self.split = split
        self.preload = preload

        self.image_path_raw = os.path.join(self.path, n_layers_path, "training_raw.txt")
        self.image_path_smooth = os.path.join(self.path, n_layers_path, "training_geo.txt")
        self.mask_path = os.path.join(self.path,n_layers_path, "training_cls.txt")

        self._get_dataset_info(os.path.join(self.shared_path,n_layers_path, "training_cls.txt"))

        self.smooth_raw_both = smooth_or_raw

        if smooth_or_raw == 'both':
            self.data_shape = (200, 2)
        else:
            self.data_shape = (200, 1)

        if self.preload:
            self.image_raw = self._preload_data(
                os.path.join(self.shared_path, n_layers_path, "training_raw.txt"),
                dtype='floatX', expand=True)
            self.image_smooth = self._preload_data(
                os.path.join(self.shared_path, n_layers_path, "training_geo.txt"),
                dtype='floatX', expand=True)
            self.mask = self._preload_data(os.path.join(self.shared_path,
                                                        n_layers_path,
                                                        "training_cls.txt"),
                                           dtype='int32')
        else:
            self.image_raw = None
            self.image_smooth = None
            self.mask = None

        super(CorticalLayersDataset, self).__init__(*args, **kwargs)

    def get_names(self):
        """Return a dict of names, per prefix/subset."""

        return {'default': self.filenames}

    def _preload_data(self, path, dtype, expand=False):
        if dtype == 'floatX':
            py_type = float
            dtype = floatX
        elif dtype == 'int32':
            py_type = int
        else:
            raise ValueError('dtype not supported', dtype)
        ret = []
        with open(path) as fp:
            for i, line in enumerate(fp):
                line = re.split(' ', line)
                line = np.array([py_type(el) for el in line], dtype=dtype)
                ret.append(line)
        ret = np.vstack(ret)
        if expand:
            # b,0 to b,0,c
            ret = np.expand_dims(ret, axis=2)
        return ret

    def _get_dataset_info(self, path):
        py_type = int

        ret = []
        with open(path) as fp:
            for i, line in enumerate(fp):
                line = re.split(' ', line)
                line = np.array([py_type(el) for el in line], dtype='int32')
                ret.append(line)
        ret = np.vstack(ret)

        self.GTclasses = list(np.sort(np.unique(ret)))
        self.non_void_nclasses = len(self.GTclasses)
        cmap = self.distinct_colours(self.non_void_nclasses)

        self._mask_labels = {}
        self._cmap = {}
        for el in self.GTclasses:
            if el == 0:
                self._mask_labels[el] = 'padding'
            else:
                self._mask_labels[el] = 'layer'+str(el)

            self._cmap[el] = tuple(cmap[el])

        return

    def distinct_colours(self, n):
       #returns n distinct colours using nipy_spectral colourmap
       np.random.seed(1)
       Range = np.round(np.linspace(1, 255, n)).astype(int)
       np.random.shuffle(Range)
       cmap = cm.nipy_spectral(Range)[:, 0:3]

       return cmap

    def load_sequence(self, sequence):
        """Load a sequence of profiles

        Auxiliary function that loads a sequence of 1D intensity profiles with
        the corresponding ground truth and their filenames.
        Returns a dict with the sequences in [0, 1], their corresponding
        labels, their subset (i.e. category, clip, prefix) and their
        filenames.
        """

        batch_to_load = [tupl[1] for tupl in sequence]
        ret = {}

        if self.smooth_raw_both=='raw' or self.smooth_raw_both=='both':
            if self.preload:
                raw = self.image_raw[batch_to_load]
            else:
                raw=[]
                with open(self.image_path_raw) as fp:
                    for i, line in enumerate(fp):
                        if i in batch_to_load:
                            line = re.split(' ', line)
                            line = np.array([float(el) for el in line])
                            line = line.astype(floatX)
                            raw.append(line)
                        if len(raw) == len(batch_to_load):
                            break
                raw = np.vstack(raw)
                # b,0 to b,0,c
                raw = np.expand_dims(raw, axis=2)

        if self.smooth_raw_both=='smooth' or self.smooth_raw_both=='both':
            if self.preload:
                smooth = self.image_smooth[batch_to_load]
            else:
                smooth=[]
                with open(self.image_path_smooth) as fp:
                    for i, line in enumerate(fp):
                        if i in batch_to_load:
                            line = re.split(' ', line)
                            line = np.array([float(el) for el in line])
                            line = line.astype(floatX)
                            smooth.append(line)
                        if len(smooth) == len(batch_to_load):
                            break

                smooth = np.vstack(smooth)
                # b,0 to b,0,c
                smooth = np.expand_dims(smooth, axis=2)

        if self.smooth_raw_both == 'raw':
            ret['data'] = raw
        elif self.smooth_raw_both == 'smooth':
            ret['data'] = smooth
        elif self.smooth_raw_both == 'both':
            ret['data'] = np.concatenate([smooth,raw],axis=2)

        ret['data'] = np.expand_dims(ret['data'],axis=2)

        # Load mask
        ret['labels'] = []
        if self.preload:
            ret['labels'] = self.mask[batch_to_load]
        else:
            with open(self.mask_path) as fp:
                for i, line in enumerate(fp):
                    if i in batch_to_load:
                        line = re.split(' ', line)
                        line = np.array([int(el) for el in line])
                        line = line.astype('int32')
                        ret['labels'].append(line)
                    if len(ret['labels']) == len(batch_to_load):
                        break
            ret['labels'] = np.vstack(ret['labels'])
        ret['labels'] = np.expand_dims(ret['labels'], axis=2)

        ret['filenames'] = batch_to_load

        ret['subset'] = 'default'

        return ret


def test():
    train_iter = CorticalLayersDataset(
        which_set='train',
        smooth_or_raw='both',
        batch_size=500,
        data_augm_kwargs={},
        return_one_hot=False,
        return_01c=False,
        return_list=True,
        use_threads=False,
        preload=True)

    valid_iter = CorticalLayersDataset(
        which_set='valid',
        smooth_or_raw = 'smooth',
        batch_size=500,
        data_augm_kwargs={},
        return_one_hot=False,
        return_01c=False,
        return_list=True,
        use_threads=False,
        preload=True)

    valid_iter2 = CorticalLayersDataset(
        which_set='valid',
        smooth_or_raw='raw',
        batch_size=500,
        data_augm_kwargs={},
        return_one_hot=False,
        return_01c=False,
        return_list=True,
        use_threads=False,
        preload=True)

    train_nsamples = train_iter.nsamples
    train_nbatches = train_iter.nbatches
    valid_nbatches = valid_iter.nbatches
    valid_nbatches2 = valid_iter2.nbatches

    # Simulate training
    max_epochs = 1
    print "Simulate training for", str(max_epochs), "epochs"
    start_training = time.time()
    for epoch in range(max_epochs):
        print "Epoch #", str(epoch)

        start_epoch = time.time()

        print "Iterate on the training set", train_nbatches, "minibatches"
        for mb in range(train_nbatches):
            start_batch = time.time()
            batch = train_iter.next()
            if mb % 5 == 0:
                print("Minibatch train {}: {} sec".format(mb, (time.time() -
                                                     start_batch)))

        print "Iterate on the validation set", valid_nbatches, "minibatches"
        for mb in range(valid_nbatches):
            start_batch = time.time()
            batch = valid_iter.next()
            if mb % 5 == 0:
                print("Minibatch valid {}: {} sec".format(mb, (time.time() -
                                                     start_batch)))

        print "Iterate on the validation set (second time)", valid_nbatches2, "minibatches"
        for mb in range(valid_nbatches2):
            start_batch = time.time()
            batch = valid_iter2.next()
            if mb % 5 == 0:
                print("Minibatch valid {}: {} sec".format(mb, (time.time() -
                                                     start_batch)))

        print("Epoch time: %s" % str(time.time() - start_epoch))
    print("Training time: %s" % str(time.time() - start_training))

if __name__ == '__main__':
    print "Loading the dataset 1 batch at a time"
    test()
    print "Success!"
