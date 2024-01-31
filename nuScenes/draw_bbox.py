import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, box_in_image

# nuscenes-devkit の初期化
nusc = NuScenes(version='v1.0-mini', dataroot='path/to/your/nuscenes/data', verbose=True)

# 例として、最初のサンプルを取得
my_sample = nusc.sample[0]

# カメラ画像を取得
camera_token = my_sample['data']['CAM_FRONT']
camera_data = nusc.get('sample_data', camera_token)
camera_filepath = nusc.get_sample_data_path(camera_token)

# カメラの内部行列を取得
calibrated_sensor = nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])
camera_intrinsic = np.array(calibrated_sensor['camera_intrinsic'])

# 画像上のバウンディングボックスを描画
fig, ax = plt.subplots()

# 画像を読み込み
image = plt.imread(camera_filepath)
ax.imshow(image)

# アノテーションのバウンディングボックスを取得し、画像上に描画
for ann_token in my_sample['anns']:
    ann_record = nusc.get('sample_annotation', ann_token)
    box = Box(ann_record['translation'], ann_record['size'], Quaternion(ann_record['rotation']))

    # バウンディングボックスをカメラの視点に変換
    box.render(ax, view=np.linalg.inv(np.array(calibrated_sensor['rotation']).reshape((3,3))), normalize=True)

plt.show()