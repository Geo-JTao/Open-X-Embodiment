# ------------------------ download dataset-------------------------------
# pip install tfds-nightly 
import tensorflow_datasets as tfds
import tqdm

# optionally replace the DATASET_NAMES below with the list of filtered datasets from the google sheet
DATASET_NAMES = ['fractal_20220817_data', 'kuka', 'bridge', 'taco_play',
 'jaco_play', 'berkeley_cable_routing', 'roboturk', 
 'nyu_door_opening_surprising_effectiveness', 'viola', 'berkeley_autolab_ur5', 
 'toto', 'language_table', 'columbia_cairlab_pusht_real', 
 'stanford_kuka_multimodal_dataset_converted_externally_to_rlds',
  'nyu_rot_dataset_converted_externally_to_rlds', 'stanford_hydra_dataset_converted_externally_to_rlds',
   'austin_buds_dataset_converted_externally_to_rlds', 'nyu_franka_play_dataset_converted_externally_to_rlds',
    'maniskill_dataset_converted_externally_to_rlds', 'furniture_bench_dataset_converted_externally_to_rlds',
     'cmu_franka_exploration_dataset_converted_externally_to_rlds', 
     'ucsd_kitchen_dataset_converted_externally_to_rlds', 
     'ucsd_pick_and_place_dataset_converted_externally_to_rlds', 
     'austin_sailor_dataset_converted_externally_to_rlds', 'austin_sirius_dataset_converted_externally_to_rlds',
      'bc_z', 'usc_cloth_sim_converted_externally_to_rlds',
       'utokyo_pr2_opening_fridge_converted_externally_to_rlds', 
       'utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds',
        'utokyo_saytap_converted_externally_to_rlds', 'utokyo_xarm_pick_and_place_converted_externally_to_rlds',
         'utokyo_xarm_bimanual_converted_externally_to_rlds', 'robo_net', 
         'berkeley_mvp_converted_externally_to_rlds', 'berkeley_rpt_converted_externally_to_rlds',
          'kaist_nonprehensile_converted_externally_to_rlds', 'stanford_mask_vit_converted_externally_to_rlds',
           'tokyo_u_lsmo_converted_externally_to_rlds', 'dlr_sara_pour_converted_externally_to_rlds', 'dlr_sara_grid_clamp_converted_externally_to_rlds', 'dlr_edan_shared_control_converted_externally_to_rlds', 'asu_table_top_converted_externally_to_rlds', 'stanford_robocook_converted_externally_to_rlds', 'eth_agent_affordances', 'imperialcollege_sawyer_wrist_cam', 'iamlab_cmu_pickup_insert_converted_externally_to_rlds', 'uiuc_d3field', 'utaustin_mutex', 'berkeley_fanuc_manipulation', 'cmu_food_manipulation', 'cmu_play_fusion', 'cmu_stretch', 'berkeley_gnm_recon', 'berkeley_gnm_cory_hall', 'berkeley_gnm_sac_son']
DOWNLOAD_DIR = '~/tensorflow_datasets'

print(f"Downloading {len(DATASET_NAMES)} datasets to {DOWNLOAD_DIR}.")
for dataset_name in tqdm.tqdm(DATASET_NAMES):
  _ = tfds.load(dataset_name, data_dir=DOWNLOAD_DIR)

# ---------------------------Data Loader Example---------------------------------
import tensorflow as tf
import tensorflow_datasets as tfds

dataset = 'kuka'
b = tfds.builder_from_directory(builder_dir=dataset2path(dataset))
ds = b.as_dataset(split='train[:10]') #只取前10个样本做实例

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

# convert RLDS episode dataset to individual steps & reformat
ds = ds.map(
    episode2steps, num_parallel_calls=tf.data.AUTOTUNE).flat_map(lambda x: x)
ds = ds.map(step_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

# shuffle, repeat, pre-fetch, batch
ds = ds.cache()         # optionally keep full dataset in memory
ds = ds.shuffle(100)    # set shuffle buffer size
ds = ds.repeat()        # ensure that data never runs out

# 对数据集批次处理，开始训练
import tqdm

for i, batch in tqdm.tqdm(
    # 提前加载3个batch放入缓存，每个batch包含4个样本
    enumerate(ds.prefetch(3).batch(4).as_numpy_iterator())):
  # here you would add your PyTorch training code
  if i == 10000: break

# ---------------------------加载不同类型数据集--------------------------------
# Load second dataset 
dataset = 'utaustin_mutex'
b = tfds.builder_from_directory(builder_dir=dataset2path(dataset))
ds2 = b.as_dataset(split='train[:10]')

def step_map_fn_mutex(step):
  # reformat to align specs of both datasets
  return {
      'observation': {
          'image': tf.image.resize(step['observation']['image'], (128, 128)),
      },
      'action': step['action'],
  }

ds2 = ds2.map(
    episode2steps, num_parallel_calls=tf.data.AUTOTUNE).flat_map(lambda x: x)
ds2 = ds2.map(step_map_fn_mutex, num_parallel_calls=tf.data.AUTOTUNE)

# shuffle, repeat, pre-fetch, batch
ds2 = ds2.cache()         # optionally keep full dataset in memory
ds2 = ds2.shuffle(100)    # set shuffle buffer size
ds2 = ds2.repeat()        # ensure that data never runs out
# interleave datasets w/ equal sampling weight
ds_combined = tf.data.Dataset.sample_from_datasets([ds, ds2], [0.5, 0.5])

import tqdm
for i, batch in tqdm.tqdm(
    enumerate(ds_combined.prefetch(3).batch(4).as_numpy_iterator())):
  if i == 10000: break
