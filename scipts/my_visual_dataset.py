import numpy as np
import tensorflow_datasets as tfds
from PIL import Image
from IPython import display

DATASETS = ['austin_buds_dataset_converted_externally_to_rlds',
            'cmu_franka_exploration_dataset_converted_externally_to_rlds',
            'utokyo_xarm_bimanual_converted_externally_to_rlds']


# 将读取的结果保存为GIF格式，并返回GIF格式数据
def as_gif(images,path='temp.gif'):
    # 将一组图片保存为GIF格式，并读取,duration:每一帧之间的时间间隔(ms)，loop:循环次数，0为一直循环
    images[0].save(path,save_all = True,append_images = images[1:],duration=100,loop=0)
    gif_bytes = open(path,'rb').read()
    return gif_bytes

if __name__=='__main__':
    dataset = DATASETS[0]
    dataset_dir = '/home/communalien/Open-X-Embodiment/dataset/{}/0.1.0'.format(dataset)
    display_key = 'image' # 用于查看features.json文件中'observation'是否含有'image'
    # tfds.builder_from_directory 函数根据 dataset_info.json下载数据并创建一个 tfds.core.DatasetBuilder
    # 数据会根据 features.json 文件内容进行解析
    builder = tfds.builder_from_directory(builder_dir=dataset_dir)
    # builder.info.features是一个大字典，字典里面嵌套字典,
    '''
    print(builder.info.features['steps'])
    Dataset({
        'action': Tensor(shape=(7,), dtype=float32),
        'discount': Scalar(shape=(), dtype=float32),
        'is_first': bool,
        'is_last': bool,
        'is_terminal': bool,
        'language_embedding': Tensor(shape=(512,), dtype=float32),
        'language_instruction': Text(shape=(), dtype=string),
        'observation': FeaturesDict({
            'image': Image(shape=(128, 128, 3), dtype=uint8),
            'state': Tensor(shape=(24,), dtype=float32),
            'wrist_image': Image(shape=(128, 128, 3), dtype=uint8),
        }),
        'reward': Scalar(shape=(), dtype=float32),
    })
    print("数据集样本数为:{}".format(len(builder.as_dataset(split='train[0:]'))))
    '''
   
    if display_key not in builder.info.features['steps']['observation']:
        raise ValueError(
            f'The key {display_key} was not found in this dataset.\n'
            + 'Please choose a different image key to display for this dataset.\n'
            + 'Here is the observation spec:\n'
            + str(builder.info.features['steps']['observation']))

    # as_dataset将一个 tfds.core.DatasetBuilder对象转换为一个tf.data.Dataset对象
    # split='train[:10]' 加载训练集的前 10 个样本，shuffle 将其打乱
    ds = builder.as_dataset(split='train[:10]').shuffle(10)
    episode = next(iter(ds)) # 对应 builder.info.features
    images = [step['observation'][display_key] for step in episode['steps']]
    images = [Image.fromarray(image.numpy()) for image in images]
    # display.Image(as_gif(images))

    for elements in next(iter(episode['steps'])).items():
        print(elements) # 具体的值
