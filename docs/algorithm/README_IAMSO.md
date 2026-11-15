# IAMSO 主程序代码说明

本文档说明当前主程序中 `IAMSO` 的实际生产实现。代码里的 `IAMSO` 菜单项对应 **IAMSO-NF-BI-CA**：无原始路径回退（NF）、双向多源种子（BI）和碰撞感知定长编码（CA）。

## 1. 主程序调用链

```text
main.py
  -> cli_inputs.select_algorithm()
  -> registry.ALGORITHMS["7"]
  -> mode_runners 调用 iamso_planner(...)
  -> build_bidirectional_seed_paths(...)
  -> collision_aware_initial_population(...)
  -> iamso_optimize(...)
  -> smooth_planner_result(...)
  -> metrics 评价并由 visualization 输出结果
```

主程序先选择运行模式、地图、起终点、算法和代价函数。选择算法编号 `7` 后，注册表返回 `iamso_planner`。模式 1-4 通过 `mode_runners/standard.py` 调用它；模式 6 也从同一注册表取得 IAMSO，但固定使用无任务价值代价函数。

## 2. 核心文件

| 文件 | 作用 |
|---|---|
| `main.py` | 交互式主入口和模式分发 |
| `uav_planner/registry.py` | 将编号 `7` 注册为 `IAMSO` |
| `uav_planner/algorithms/iamso/planner.py` | IAMSO 适配层：生成种子、设置权重、调用优化器、整理返回路径 |
| `uav_planner/algorithms/iamso/seeds.py` | 双向网格搜索及安全 RRT/RRT* 种子 |
| `uav_planner/metaheuristic/collision_aware.py` | 碰撞感知种子压缩和初始种群构造 |
| `uav_planner/metaheuristic/iamso.py` | 自适应权重、适应度、Superior/Inferior 双阶段搜索 |
| `uav_planner/metaheuristic/feasibility.py` | 可行性优先排序和最优可行解档案 |
| `uav_planner/metaheuristic/encoding.py` | 航点向量编解码、采样和 Catmull-Rom 平滑 |
| `uav_planner/geometry.py` | 连续线段碰撞检测及规划器结果封装 |

## 3. 入口接口

```python
iamso_planner(passable, speed_map, cost_func, start, goal)
```

| 参数 | 含义 |
|---|---|
| `passable` | 方形二维可通行矩阵，真值/非零表示可通行 |
| `speed_map` | 当前地图层数据；主程序通常传入包含风场、降雨、巡航价值等信息的分层地图 |
| `cost_func` | 当前实验选择的代价函数，用于构造服从当前目标的网格种子 |
| `start`、`goal` | `(row, col)` 形式的起点和终点 |

返回值是列表兼容的 `PathResult`，路径点为浮点 `(row, col)`。它还携带：

- `solution_source`：解来自初始种群、优化过程或不可行最优解；
- `feasible`：优化器输出平滑前的路径是否无碰撞；
- `seed_labels`：本次成功生成的种子类型；
- `used_fallback`：生产 IAMSO 固定为 `False`。

如果优化器没有路径则返回 `None`。如果没有找到可行解，优化器仍会返回当前最优不可行路径，并令 `feasible=False`，供统一指标代码记录为碰撞失败。

## 4. 双向多源种子

`build_bidirectional_seed_paths` 最多生成 8 条种子：

| 标签 | 方法 |
|---|---|
| `bidirectional_astar_current` | 双向 A* |
| `bidirectional_distance_field_astar_current` | 加入障碍距离场引导的双向 A* |
| `bidirectional_theta_star_current` | 带视线松弛的双向 Theta* |
| `bidirectional_dijkstra_current` | 双向 Dijkstra |
| `bidirectional_dynamic_weighted_astar_e1` | `epsilon=1.0` 的动态加权双向 A* |
| `bidirectional_dynamic_weighted_astar_e2` | `epsilon=2.0` 的动态加权双向 A* |
| `safe_rrt` | 每条树边都做连续碰撞检测的 RRT |
| `safe_rrt_star` | 带近邻择优和重连的安全 RRT* |

网格种子的累计代价使用当前 `cost_func`，因此会随主程序所选优化目标变化。RRT/RRT* 用于补充几何多样性。所有候选种子进入种群前都必须通过连续路径碰撞检查；生成失败或发生碰撞的种子直接跳过。

随机过程使用由起终点和固定偏移计算的种子，因此相同输入可重复。

## 5. 定长编码与初始种群

默认参数为：

| 参数 | 默认值 |
|---|---:|
| 种群规模 `population_size` | 24 |
| 迭代次数 `iterations` | 80 |
| 中间航点数 `waypoint_count` | 6 |
| 决策向量维数 | 12 |

一个个体表示为：

```text
[r1, c1, r2, c2, ..., r6, c6]
```

起点和终点不进入决策向量，解码时固定添加在两端。所有坐标都裁剪到 `[0, map_size-1]`。

碰撞感知编码按以下顺序处理种子路径：

1. 用视线检测删除不必要点，同时保留绕障所需拐点。
2. 若简化后仍超过“6 个中间点 + 2 个端点”，放弃该种子，避免压缩后穿越障碍。
3. 若点数不足，在最长线段中反复插入中点，补齐固定长度。

初始种群包含直线个体、可编码的多源种子、种子邻域高斯扰动个体和随机个体。前 12 个槽位以内优先放置种子及其扰动，其余保持随机，用于维持搜索多样性。

## 6. 自适应环境权重

普通模式的基础权重为：

```text
(障碍膨胀, 风场, 降雨, 巡航价值) = (0.35, 0.20, 0.25, 0.20)
```

模式 6 的无任务基础权重为 `(0.35, 0.20, 0.25, 0.00)`。优化器根据地图统计量调整权重：

```text
raw = base * (1 + [障碍占比, 平均风速, 平均降雨, 平均巡航价值])
weights = raw / sum(raw)
```

在分层地图上，IAMSO 搜索阶段直接使用这组自适应综合权重计算环境代价。此时主程序传入的 `cost_func` 主要影响多源种子的生成；在非分层兼容输入上，适应度评价才直接调用 `cost_func`。

## 7. 路径适应度

每个决策向量先解码为折线路径，再用 Catmull-Rom 曲线平滑，并以 `0.5` 的步长采样。默认最小化目标为：

```text
J = J_layer
  + 1.0 * L
  + 1000.0 * N_segment_collision
  + 2.0 * S_turn
  + 1.0 * E_exposure
```

其中：

- `J_layer`：沿路径积分得到的障碍膨胀、风场、降雨和巡航综合代价；
- `L`：路径欧氏长度；
- `N_segment_collision`：与障碍相交的路径线段数；
- `S_turn`：各转角弧度平方之和；
- `E_exposure`：连续降雨区和障碍膨胀区的暴露风险。

适应度中的碰撞惩罚用于拉开不可行解，但候选选择并不只比较标量 `J`。

## 8. 可行性优先规则

IAMSO 使用以下字典序比较键，值越小越好：

```text
(是否不可行, 碰撞线段数, 碰撞单元数, J, 平滑度)
```

因此：

1. 任意无碰撞路径都优于有碰撞路径，即使它的标量适应度更大。
2. 不可行路径先减少碰撞线段，再减少碰撞单元，最后比较适应度。
3. `FeasibleArchive` 独立保存搜索过程中出现过的最佳无碰撞路径，后续种群更新不会将其丢失。

这是当前 IAMSO 保证“成功率优先”的核心机制。

## 9. Superior/Inferior 双阶段搜索

每轮迭代都先执行 Superior 阶段，再视进度执行 Inferior 阶段。

### 9.1 Superior 阶段

- 迭代前 40%：结合全局最优个体、种群均值、适应度-距离平衡个体和正弦扰动做广域探索。
- 迭代后 60%：使用全局最优、随机同伴和逐渐衰减的高斯噪声做引导搜索。
- 适应度-距离个体按 `65%` 适应度质量与 `35%` 到最优个体的距离选出。

### 9.2 Inferior 阶段

- 前 15% 迭代不执行，先保留探索空间。
- 之后将个体拉向全局最优，并按个体间距离加入随迭代收缩的局部扰动。

### 9.3 停滞反馈

算法比较最近 7 轮的最优值：

- 改进不足时，扰动因子设为 `1.5`，扩大搜索；
- 仍有明显改进时，扰动因子设为 `0.75`，收紧搜索。

每个候选都先裁剪到地图边界，再按可行性优先键与原个体比较。候选不差于原个体时才即时替换。

## 10. 最终解与后处理

搜索结束后：

1. 若可行解档案非空，返回档案中的最佳无碰撞路径。
2. 否则返回当前最优不可行路径，并标记 `solution_source="best_infeasible"`。
3. `iamso_planner` 再调用 `smooth_planner_result(..., samples_per_segment=10)` 生成最终展示路径。
4. 统一运行器重新计算距离、时间、安全度、碰撞、平滑度、暴露和综合评分。

需要注意：适配层的最后一次平滑不执行碰撞修复，也不会更新 `PathResult.feasible`。因此对最终输出路径，应以统一运行器重新计算的碰撞指标和状态为准。

`iamso_optimize` 支持 `fallback_path`，但当前生产入口明确传入 `None`，即 NF。`planner.py` 中的 `_seed_paths`、`_cardinal_safe_path` 和 `_safe_fallback_path` 是兼容/实验辅助代码，不在当前 `iamso_planner` 调用链中。

## 11. 伪代码

```text
输入 passable, layers, cost_func, start, goal
生成双向 A*/Theta*/Dijkstra/动态加权 A* 和安全 RRT/RRT* 种子
过滤碰撞种子
用碰撞感知方式压缩种子为 6 个中间航点
构造 24 个个体的初始种群
计算地图自适应权重
评价种群并初始化全局最优与可行解档案

重复 80 轮：
    计算停滞自适应因子
    对全部个体执行 Superior 更新和可行性优先替换
    若迭代比例 >= 0.15：
        对全部个体执行 Inferior 更新和可行性优先替换
    记录当前最优适应度

若存在历史可行解：返回最佳可行解
否则：返回当前最优不可行解
对返回路径做展示平滑并交给统一指标模块评价
```

## 12. 运行方式

```powershell
uv sync
.\.venv\Scripts\python.exe .\main.py
```

单次运行 IAMSO：选择模式 `1`，再选择地图参数、算法 `7` 和所需代价函数。固定地图实验：选择模式 `6`、单一算法实验，再选择算法 `7`。

## 13. 最小回归检查

```powershell
.\.venv\Scripts\python.exe -m unittest `
  tests.test_iamso_feasibility `
  tests.test_collision_aware_compression `
  tests.test_iamso_seed_ablation
```

这些测试覆盖默认搜索预算、NF-BI-CA 生产入口、可行性优先排序、可行解档案、碰撞感知压缩和种子变体。
