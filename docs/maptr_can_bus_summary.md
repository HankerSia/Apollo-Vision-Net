# MapTR 中 CAN 数据的使用总结

MapTR 里的 CAN 数据不是用来直接生成地图真值的，而是作为时序建模和 BEV 对齐的辅助信息。
默认流程中，MapTR 只读取 CAN 扩展包里的 `pose` 通道，并把它转换成一个 18 维的 `can_bus` 向量。
随后，这个向量会在数据集阶段被重写为当前帧位姿，再在 Transformer 中用于 BEV 平移补偿、历史 BEV 旋转对齐和 query 条件注入。

## 1. 使用了哪些 CAN 数据

默认代码路径里，MapTR 只主动使用 `pose`。
下面这些消息虽然在 nuScenes CAN 扩展包里存在，但不进入 MapTR 主训练链路：

- `ms_imu`
- `route`
- `steeranglefeedback`
- `vehicle_monitor`
- `zoesensors`
- `zoe_veh_info`

换句话说，MapTR 依赖的是 `pose` 提供的位姿与运动状态，而不是整套 CAN 原始消息。

## 2. 18 维 can_bus 的生成方式

数据转换阶段会按 sample 的时间戳，从当前 scene 的 `pose` 序列里找到不晚于该 sample 的最后一条记录，然后把它拼成 18 维向量。

18 维的来源可以写成：

```text
can_bus[0:3] = pos
can_bus[3:7] = orientation
can_bus[7:10] = accel
can_bus[10:13] = rotation_rate
can_bus[13:16] = vel
can_bus[16:18] = 两个占位角度值，后续在数据集里会被重写
```

对应的核心代码在 `tools/data_converter/nuscenes_converter.py`：

```python
def _get_can_bus_info(nusc, nusc_can_bus, sample):
    scene_name = nusc.get('scene', sample['scene_token'])['name']
    sample_timestamp = sample['timestamp']
    try:
        pose_list = nusc_can_bus.get_messages(scene_name, 'pose')
    except:
        return np.zeros(18)  # server scenes do not have can bus information.
    can_bus = []
    # during each scene, the first timestamp of can_bus may be large than the first sample's timestamp
    last_pose = pose_list[0]
    for i, pose in enumerate(pose_list):
        if pose['utime'] > sample_timestamp:
            break
        last_pose = pose
    _ = last_pose.pop('utime')  # useless
    pos = last_pose.pop('pos')
    rotation = last_pose.pop('orientation')
    can_bus.extend(pos)
    can_bus.extend(rotation)
    for key in last_pose.keys():
        can_bus.extend(pose[key])  # 16 elements
    can_bus.extend([0., 0.])
    return np.array(can_bus)
```

从你本地 `scene-0103_pose.json` 的字段结构来看，单条 `pose` 消息包含：

- `pos`
- `orientation`
- `accel`
- `rotation_rate`
- `vel`
- `utime`

因此，这个 18 维向量的拼接逻辑是明确的：先取位姿四元数和平移，再补上加速度、角速度、速度，最后补两个占位值。

## 3. 数据集里怎么重写 can_bus

进入数据集后，MapTR 不再把这 18 维理解为原始 CAN 语义，而是直接重写成当前帧的全局位姿和朝向。

相关代码在 `projects/mmdet3d_plugin/datasets/nuscenes_map_dataset.py`：

```python
rotation = Quaternion(input_dict['ego2global_rotation'])
translation = input_dict['ego2global_translation']
can_bus = input_dict['can_bus']
can_bus[:3] = translation
can_bus[3:7] = rotation
patch_angle = quaternion_yaw(rotation) / np.pi * 180
if patch_angle < 0:
    patch_angle += 360
can_bus[-2] = patch_angle / 180 * np.pi
can_bus[-1] = patch_angle
```

这一段的含义是：

- 前 3 维变成当前帧 `ego2global_translation`
- 第 4 到 7 维变成当前帧 `ego2global_rotation`
- 倒数第 2 维写成朝向角的弧度形式
- 最后 1 维写成朝向角的角度形式

## 4. 时序队列里怎么用

如果启用了多帧输入，MapTR 会把 `can_bus` 变成相邻帧的相对增量。
第一帧清零，后续帧则减去上一帧的位移和角度。

```python
prev_pos = None
prev_angle = None
for i, each in enumerate(queue):
    metas_map[i] = each['img_metas'].data
    if i == 0:
        metas_map[i]['prev_bev'] = False
        prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
        prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
        metas_map[i]['can_bus'][:3] = 0
        metas_map[i]['can_bus'][-1] = 0
    else:
        metas_map[i]['prev_bev'] = True
        tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
        tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
        metas_map[i]['can_bus'][:3] -= prev_pos
        metas_map[i]['can_bus'][-1] -= prev_angle
        prev_pos = copy.deepcopy(tmp_pos)
        prev_angle = copy.deepcopy(tmp_angle)
```

这一步的作用是让模型看到的是“相对运动”，而不是绝对位姿。

## 5. Transformer 里怎么消费

最终 `can_bus` 会进入 Transformer，主要做三件事：

1. 根据平移量计算 BEV shift
2. 根据角度旋转上一帧 `prev_bev`
3. 把完整 `can_bus` 经过 MLP 后加到 BEV queries 上

相关代码在 `projects/mmdet3d_plugin/maptr/modules/transformer.py`：

```python
delta_x = np.array([each['can_bus'][0]
                   for each in kwargs['img_metas']])
delta_y = np.array([each['can_bus'][1]
                   for each in kwargs['img_metas']])
ego_angle = np.array(
    [each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])
grid_length_y = grid_length[0]
grid_length_x = grid_length[1]
translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
bev_angle = ego_angle - translation_angle
shift_y = translation_length * \
    np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
shift_x = translation_length * \
    np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w

if prev_bev is not None:
    if self.rotate_prev_bev:
        for i in range(bs):
            rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
            tmp_prev_bev = prev_bev[:, i].reshape(
                bev_h, bev_w, -1).permute(2, 0, 1)
            tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                  center=self.rotate_center)
            prev_bev[:, i] = tmp_prev_bev.permute(1, 2, 0).reshape(
                bev_h * bev_w, 1, -1)[:, 0]

can_bus = bev_queries.new_tensor(
    [each['can_bus'] for each in kwargs['img_metas']])
can_bus = self.can_bus_mlp(can_bus)[None, :, :]
bev_queries = bev_queries + can_bus * self.use_can_bus
```

## 6. Map 真值不是 CAN 生成的

MapTR 的 map 真值来自 nuScenes 高精地图和当前帧位姿，不依赖 CAN 的其他消息。
离线 map annotation 的生成入口在 `tools/maptrv2/custom_nusc_map_converter.py`：

```python
def obtain_vectormap(nusc_maps, map_explorer, info, point_cloud_range):
    lidar2ego = np.eye(4)
    lidar2ego[:3,:3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
    lidar2ego[:3, 3] = info['lidar2ego_translation']
    ego2global = np.eye(4)
    ego2global[:3,:3] = Quaternion(info['ego2global_rotation']).rotation_matrix
    ego2global[:3, 3] = info['ego2global_translation']

    lidar2global = ego2global @ lidar2ego

    lidar2global_translation = list(lidar2global[:3,3])
    lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)

    location = info['map_location']
    patch_h = point_cloud_range[4]-point_cloud_range[1]
    patch_w = point_cloud_range[3]-point_cloud_range[0]
    patch_size = (patch_h, patch_w)
    vector_map = VectorizedLocalMap(nusc_maps[location], map_explorer[location],patch_size)
    map_anns = vector_map.gen_vectorized_samples(lidar2global_translation, lidar2global_rotation)
    info["annotation"] = map_anns
    return info
```

这说明真值生成的核心输入是：

- 地图位置 `map_location`
- 当前帧位姿 `ego2global_translation`
- 当前帧位姿 `ego2global_rotation`

不是 CAN 的其他消息内容。

## 7. 小结

MapTR 里 CAN 数据的职责很明确：它提供车辆运动和位姿的辅助表达，帮助模型做时序 BEV 对齐和 query 条件编码。
默认只使用 `pose`，并把它整理成 18 维 `can_bus`；随后在数据集里重写为当前帧位姿，在队列里改成相对运动，在 Transformer 里用于 BEV shift、历史 BEV 旋转和特征注入。

如果只看 map 真值生成，核心依赖其实是 nuScenes 高精地图和 ego 位姿，而不是整套 CAN 原始消息。