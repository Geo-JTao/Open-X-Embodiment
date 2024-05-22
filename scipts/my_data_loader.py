import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
from IPython import display
import tqdm
# 不同的数据集不一样的维度：'action': TensorSpec(shape=(7,)，(8,)，(14,)
DATASETS = [
            # 'austin_buds_dataset_converted_externally_to_rlds',
            'cmu_franka_exploration_dataset_converted_externally_to_rlds',
            # 'utokyo_xarm_bimanual_converted_externally_to_rlds'
           ]

def episode2steps(episode):
  return episode['steps']

# 预处理操作使原始数据适合训练
def step_map_fn(step):
  return {
      'observation': {
          'image': tf.image.resize(step['observation']['image'], (128, 128)),
      },
      'action': tf.concat([
          step['action']['world_vector'],
          step['action']['rotation_delta'],
          step['action']['gripper_closedness_action'],
      ], axis=-1)
  }

# 不同数据集混合处理
def step_map_fn_mutex(step):
  return {
      'observation': {
          'image': tf.image.resize(step['observation']['image'], (128, 128)),
      },
      'action': step['action'],
  }

if __name__=='__main__':

    datasets = []
    for index, dataset_name in enumerate(DATASETS):
        dataset_dir = '/home/communalien/Open-X-Embodiment/dataset/{}/0.1.0'.format(dataset_name)
        builder = tfds.builder_from_directory(builder_dir=dataset_dir)
        dataset = builder.as_dataset(split='train[:10]')
        dataset = dataset.map(episode2steps, num_parallel_calls=tf.data.AUTOTUNE).flat_map(lambda x: x)
        dataset = dataset.map(step_map_fn_mutex, num_parallel_calls=tf.data.AUTOTUNE)
        # shuffle, repeat, pre-fetch, batch
        dataset = dataset.cache().shuffle(10).repeat()
        datasets.append(dataset)
    # 混合加载多个数据集，权重为均等概率
    weights = [1/len(datasets)] * len(datasets)
    dataset_combined = tf.data.Dataset.sample_from_datasets(datasets, weights)
    print(dataset_combined)
    for i, batch in tqdm.tqdm(
        # 提前加载3个batch放入缓存，每个batch包含4个样本
        enumerate(dataset_combined.prefetch(3).batch(4).as_numpy_iterator())):
    # here you would add your PyTorch training code
        if i == 10000: break


