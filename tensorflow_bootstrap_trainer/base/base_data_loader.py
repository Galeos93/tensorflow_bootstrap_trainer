class BaseDataLoader(object):
    def __init__(self, config):
        self.config = config

    def get_train_data(self):
        raise NotImplementedError

    def get_test_data(self):
        raise NotImplementedError

import numpy as np
import tensorflow as tf

def transformer(image, transformation=lambda x: {"image": x}):
    data = {"image": image}
    aug_data = transformation(**data)
    aug_img = aug_data["image"]
    return aug_img

def transform_data(image, label, transformer):
    aug_img = tf.numpy_function(func=transformer, inp=[image], Tout=tf.float64)
    return aug_img, label


class DatasetFactory:
    dataset = wds.WebDataset("/mnt/agarcia_HDD/LFW/LFW-test.tar").decode("rgb").to_tuple("jpg;png", "json")
    
    @classmethod
    def _generator(cls):
        # Opening the file        
        for image, meta in cls.dataset:
            yield image, meta["image_num"] 
    
    @classmethod
    def create_in_memory(cls):
        tensor_slices = tuple(zip(*cls._generator()))
        tensor_slices = tuple(np.array(item) for item in tensor_slices)
        return tf.data.Dataset.from_tensor_slices(tensor_slices)
        
    @classmethod
    def create_sequence(cls):       
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.float64, tf.int64),
            output_shapes=((250, 250, 3), ()),
        )

