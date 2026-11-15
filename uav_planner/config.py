"""
全局配置与常量
"""

import math

# 地图尺寸
SIZE = 100

# 正常飞行速度
NORMAL_SPEED = 3
MIN_SPEED_FACTOR = 0.20
MAX_TAILWIND_SPEED_FACTOR = 1.30
WIND_SPEED_EFFECT = 0.50

# 八方向移动，第三项为对应步长。
MOVES = [
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)),
    (-1, 1, math.sqrt(2)),
    (-1, -1, math.sqrt(2)),
]

# 多层代价地图参数
RHO = 0.35
R_INF = 12.0
V_MAX = 1.0
GAMMA = 1.5
SIGMA = 8.0
M_TASKS = 5
N_OBSTACLES = 18
OBSTACLE_SIZE_RANGE = (4, 16)
OBSTACLE_TARGET_RATIO = {
    30: 0.15,
    60: 0.25,
    90: 0.35,
}

# 模型代价权重：(膨胀区, 风场, 降雨, 巡航价值)
MODEL_WEIGHTS = (0.35, 0.20, 0.25, 0.20)

# Planning-stage model cost weights.
MODEL_COST_WEIGHTS = {
    "length": 1.0,
    "inflation": 0.35,
    "wind": 0.20,
    "rain": 0.25,
    "cruise": 0.20,
}

# Nonlinear exposure-risk parameters.
EXPOSURE_INTENSITY_GAMMA = 1.5
EXPOSURE_LENGTH_BETA = 1.25
EXPOSURE_RAIN_THRESHOLD = 0.05
EXPOSURE_INFLATION_THRESHOLD = 0.05
EXPOSURE_PATH_WEIGHT = 1.0

# Evaluation-stage score weights. Cost-like terms are converted to quality terms;
# task reward is a positive term. Final score is 0-100 and higher is better.
EVALUATION_SCORE_WEIGHTS = {
    "length": 0.20,
    "time": 0.20,
    "risk": 0.20,
    "collision": 0.15,
    "smoothness": 0.10,
    "runtime": 0.00,
    "task": 0.10,
}

# No-task corner benchmark profile. It keeps the environmental model but removes
# cruise/task value from planning and scoring.
NO_TASK_MODEL_COST_WEIGHTS = {
    "length": 1.0,
    "inflation": 0.35,
    "wind": 0.20,
    "rain": 0.25,
    "cruise": 0.0,
}

NO_TASK_EVALUATION_SCORE_WEIGHTS = {
    "length": 0.22,
    "time": 0.22,
    "risk": 0.24,
    "collision": 0.18,
    "smoothness": 0.14,
    "runtime": 0.00,
    "task": 0.00,
}
