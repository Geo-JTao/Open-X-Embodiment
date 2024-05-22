import tensorflow as tf
import tensorflow_datasets as tfds
import rlds
from PIL import Image
import numpy as np
from tf_agents.policies import py_tf_eager_policy
import tf_agents
from tf_agents.trajectories import time_step as ts
from IPython import display
from collections import defaultdict
import matplotlib.pyplot as plt

DATASETS = ['austin_buds_dataset_converted_externally_to_rlds',
            'cmu_franka_exploration_dataset_converted_externally_to_rlds',
            'utokyo_xarm_bimanual_converted_externally_to_rlds',
            'bridge',]

def as_gif(images):
  images[0].save('temp.gif', save_all=True, append_images=images[1:], duration=1000, loop=0)
  gif_bytes = open('temp.gif','rb').read()
  return gif_bytes

# 将图像转为[320，256]，uint8类型
def resize(image):
  image = tf.image.resize_with_pad(image, target_width=320, target_height=256)
  image = tf.cast(image, tf.uint8)
  return image

# 判断布尔值terminate_episode是否为1，如果为1表示终止返回动作向量[1，0，0]，否则[0,1,0]
def terminate_bool_to_act(terminate_episode: tf.Tensor) -> tf.Tensor:
  return tf.cond(
      terminate_episode == tf.constant(1.0),
      lambda: tf.constant([1, 0, 0], dtype=tf.int32),
      lambda: tf.constant([0, 1, 0], dtype=tf.int32),
  )

# 将输入动作缩放到指定范围[-1,1]内
def rescale_action_with_bound(
    actions: tf.Tensor,
    low: float,
    high: float,
    safety_margin: float = 0,
    post_scaling_max: float = 1.0,
    post_scaling_min: float = -1.0,
) -> tf.Tensor:
  """Formula taken from https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range."""
  resc_actions = (actions - low) / (high - low) * (
      post_scaling_max - post_scaling_min
  ) + post_scaling_min
  return tf.clip_by_value(
      resc_actions,
      post_scaling_min + safety_margin,
      post_scaling_max - safety_margin,
  )

def rescale_action(action):
  """Rescales action."""
  action['world_vector'] = rescale_action_with_bound(
      action['world_vector'],
      low=-0.05,
      high=0.05,
      safety_margin=0.01,
      post_scaling_max=1.75,
      post_scaling_min=-1.75,
  )
  action['rotation_delta'] = rescale_action_with_bound(
      action['rotation_delta'],
      low=-0.25,
      high=0.25,
      safety_margin=0.01,
      post_scaling_max=1.4,
      post_scaling_min=-1.4,
  )
  return action

def to_model_action(from_step):
  """Convert dataset action to model action. This function is specific for the Bridge dataset."""

  model_action = {}

  model_action['world_vector'] = from_step['action']['world_vector']
  model_action['terminate_episode'] = terminate_bool_to_act(
      from_step['action']['terminate_episode']
  )
  model_action['rotation_delta'] = from_step['action']['rotation_delta']
  open_gripper = from_step['action']['open_gripper']

  # 创建possible_values包含[True, False]
  possible_values = tf.constant([True, False], dtype=tf.bool)
  # 确保open_gripper为True或False，
  eq = tf.equal(possible_values, open_gripper)
  assert_op = tf.Assert(tf.reduce_any(eq), [open_gripper])

  with tf.control_dependencies([assert_op]):
    # 根据open_gripper是True或False设置夹爪动作值
    model_action['gripper_closedness_action'] = tf.cond(
        # for open_gripper in bridge dataset, 0 is fully closed and 1 is fully open
        open_gripper,
        # for Fractal data, -1 means opening the gripper and 1 means closing the gripper.
        lambda: tf.constant([-1.0], dtype=tf.float32),
        lambda: tf.constant([1.0], dtype=tf.float32),
    )

  model_action = rescale_action(model_action)

  return model_action

if __name__ == '__main__':
    saved_model_path = '/home/communalien/Open-X-Embodiment/rt_1_x_tf_trained_for_002272480_step'

    # 创建tfa_policy策略对象 .pbtxt描述模型结构和参数
    tfa_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
        model_path=saved_model_path,
        load_specs_from_pbtxt=True,
        use_tf_function=True)

    # 使用特征都为0的虚拟观测值执行一步推理
    observation = tf_agents.specs.zero_spec_nest(tf_agents.specs.from_spec(tfa_policy.time_step_spec.observation))
    # 将观测值转换成一个时间步对象（包含观测、奖励等必要信息）
    tfa_time_step = ts.transition(observation, reward=np.zeros((), dtype=np.float32))
    # 初始化policy状态，仅处理单个样本
    policy_state = tfa_policy.get_initial_state(batch_size=1)
    # 依据时间步和策略状态得到动作
    action = tfa_policy.action(tfa_time_step, policy_state)

    # 使用真实数据集,获取steps
    dataset = DATASETS[3]
    dataset_dir = '/home/communalien/Open-X-Embodiment/dataset/{}/0.1.0'.format(dataset)
    builder = tfds.builder_from_directory(builder_dir=dataset_dir)
    print('step数据:{}'.format(builder.info.features['steps']))
    ds = builder.as_dataset(split='train[:1]')
    ds_iterator = iter(ds)
    # Obtain the steps from one episode from the dataset
    episode = next(ds_iterator)
    steps = episode[rlds.STEPS]

    images = []
    for step in steps:
        im = Image.fromarray(np.array(step['observation']['image']))
        images.append(im)

    print(f'{len(images)} images')
    display.Image(as_gif(images))

    # 逐步处理，提取图像数据，选择执行动作
    steps = list(steps)
    print(f'{len(steps)} steps')
    policy_state = tfa_policy.get_initial_state(batch_size=1)

    gt_actions = []
    predicted_actions = []
    images = []

    for step in steps:

        image = resize(step[rlds.OBSERVATION]['image'])

        images.append(image)
        observation['image'] = image
        # 使用观测到的图片数据和零奖励创建一个时间步对象
        tfa_time_step = ts.transition(observation, reward=np.zeros((), dtype=np.float32))
        # 模型根据当前状态选择的动作,某一步的策略对象policy_step
        policy_step = tfa_policy.action(tfa_time_step, policy_state)
        action = policy_step.action
        policy_state = policy_step.state

        predicted_actions.append(action)
        # 将step数据转换成模型所需的动作格式,用于记录真实的动作数据
        gt_actions.append(to_model_action(step))

    # 创建两个defaultdict类型空字典
    action_name_to_values_over_time = defaultdict(list)
    predicted_action_name_to_values_over_time = defaultdict(list)

    # 定义figure_layout和action_order表示要记录的动作名称、维度顺序
    figure_layout = ['terminate_episode_0', 'terminate_episode_1',
            'terminate_episode_2', 'world_vector_0', 'world_vector_1',
            'world_vector_2', 'rotation_delta_0', 'rotation_delta_1',
            'rotation_delta_2', 'gripper_closedness_action_0']
    action_order = ['terminate_episode', 'world_vector', 'rotation_delta', 'gripper_closedness_action']

    # 遍历数据，将真实动作和预测动作按照指定顺序和维度记录在相应字典中
    for i, action in enumerate(gt_actions):

        for action_name in action_order:

            for action_sub_dimension in range(action[action_name].shape[0]):

                # print(action_name, action_sub_dimension)
                title = f'{action_name}_{action_sub_dimension}'

                action_name_to_values_over_time[title].append(action[action_name][action_sub_dimension])
                predicted_action_name_to_values_over_time[title].append(predicted_actions[i][action_name][action_sub_dimension])
    
    # 定义二维列表用作可视化
    figure_layout = [
                        ['image'] * len(figure_layout),
                        figure_layout
                    ]

    # 使用matplotlib绘图,设置字体大小          
    plt.rcParams.update({'font.size': 12})

    # 将图像按axis=0轴拆分得到一个图像数据列表,再按水平方向axis=1合并
    stacked = tf.concat(tf.unstack(images[::3], axis=0), 1)
    # 创建多个子图，设置尺寸
    fig, axs = plt.subplot_mosaic(figure_layout)
    fig.set_size_inches([45, 10])

    # 循环遍历字典中的键值对
    for i, (k, v) in enumerate(action_name_to_values_over_time.items()):

        # 绘制真实动作、预测动作的曲线图
        axs[k].plot(v, label='ground truth')
        axs[k].plot(predicted_action_name_to_values_over_time[k], label='predicted action')

        # 根据存储action_order内的数据名称确定标题
        axs[k].set_title(k)
        axs[k].set_xlabel('Time in one episode')

    axs['image'].imshow(stacked.numpy())
    axs['image'].set_xlabel('Time in one episode (subsampled)')
    plt.legend()
  