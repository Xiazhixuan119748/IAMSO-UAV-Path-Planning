# 有任务价值固定基准实验

本模块保存固定地图数据、批量评估逻辑和 IAMSO 专项回归脚本。它与普通随机地图模式、模式 6 无任务价值实验相互独立。

## 目录

```text
fixed_benchmark/
├── dataset.py                    数据集定义和读写
├── map_generator.py              固定基准地图生成
├── runner.py                     批量评估与报告输出
├── generate_maps.py              地图生成入口
├── run_benchmark.py              全算法基线入口
├── reachability_probe.py         可达率评估逻辑
├── run_reachability_probe.py
├── run_reachability_ratio_candidates.py
├── run_iamso_collision_weight_regression.py
├── run_mso_ao_no_exposure_vs_iamso.py
├── maps/                         当前 90 张固定地图
├── datasets/                     专项回归数据集
└── results/                      当前输出与历史归档
```

## 当前地图集

数据集版本：`fixed_benchmark_maps v3.0`。

| 难度 | 数量 | 尺寸 | 目标障碍占比 | 起终点安全半径 |
|---|---:|---:|---:|---:|
| simple | 30 | 30x30 | 15% | 2 |
| medium | 30 | 60x60 | 25% | 3 |
| hard | 30 | 90x90 | 35% | 4 |

地图具有以下约束：

- 起点固定在地图中心。
- 终点固定在右上象限并保留边界距离。
- 使用多形状障碍，不生成墙状障碍。
- 放置障碍时保护起点到终点的连通性。
- 保存完整地图层，而不只保存随机种子。

## 常用命令

从项目根目录运行。

生成或重建地图：

```powershell
.\.venv\Scripts\python.exe -m uav_planner.experiments.fixed_benchmark.generate_maps
.\.venv\Scripts\python.exe -m uav_planner.experiments.fixed_benchmark.generate_maps --force
```

运行全部算法基线：

```powershell
.\.venv\Scripts\python.exe -m uav_planner.experiments.fixed_benchmark.run_benchmark
.\.venv\Scripts\python.exe -m uav_planner.experiments.fixed_benchmark.run_benchmark --difficulty simple
```

默认使用注册表中的模型综合代价。可通过 `--cost-key 1|2|3|4` 切换代价函数。

## 历史 hard35 碰撞权重专项回归

`datasets/iamso_collision_weight_hard35_reachable_100/` 仅包含 100 张 `90x90`、目标障碍占比 35% 的可达地图，用于历史碰撞权重消融和 MSO/AO 对照实验。它不是当前模式 6 的三档活动地图集。

```powershell
.\.venv\Scripts\python.exe -m uav_planner.experiments.fixed_benchmark.run_iamso_collision_weight_regression
.\.venv\Scripts\python.exe -m uav_planner.experiments.fixed_benchmark.run_iamso_collision_weight_regression --force-dataset
.\.venv\Scripts\python.exe -m uav_planner.experiments.fixed_benchmark.run_mso_ao_no_exposure_vs_iamso
.\.venv\Scripts\python.exe -m uav_planner.experiments.fixed_benchmark.run_mso_ao_no_exposure_vs_iamso --disable-exposure
```

## 可达率评估

```powershell
.\.venv\Scripts\python.exe -m uav_planner.experiments.fixed_benchmark.run_reachability_probe
.\.venv\Scripts\python.exe -m uav_planner.experiments.fixed_benchmark.run_reachability_ratio_candidates
```

这些命令将报告写入 `results/`，不会修改普通模式或模式 6 的地图。整理前的历史报告保存在 `results/archive/`。

## NPZ 数据格式

每张地图包含：

- `O`：障碍层。
- `I`：障碍膨胀层。
- `W`：风场向量层。
- `R`：降雨层。
- `V`：巡航价值层。
- `speed_map`、`passable`、`tasks`。
- `start`、`goal`、`difficulty`、`map_size`、`seed`。
- `target_obstacle_ratio`、`safety_radius`。

活动数据集总索引位于 `maps/index.json`。历史结果中的旧路径字段用于记录当时运行环境，不作为当前加载路径。
