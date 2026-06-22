"""
config.py -- 全局配置中心

API Key 安全策略：
  · 优先读取系统环境变量（DEEPSEEK_API_KEY / DASHSCOPE_API_KEY）
  · 其次读取项目根目录 .env 文件（需安装 python-dotenv）
  · config.py 中不再硬编码任何 Key
"""
import os

# 尝试加载 .env 文件
_ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
try:
    from dotenv import load_dotenv
    if os.path.exists(_ENV_FILE):
        load_dotenv(_ENV_FILE, override=True)
except ImportError:
    if os.path.exists(_ENV_FILE):
        # python-dotenv 未安装但 .env 存在：手动解析
        with open(_ENV_FILE, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _key, _, _val = _line.partition("=")
                    _key = _key.strip()
                    _val = _val.strip().strip('"').strip("'")
                    if _key and _val:
                        os.environ.setdefault(_key, _val)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== 模型路径 =====
# 本地权重默认存放路径（开发机约定）；通过环境变量覆盖即可在他机运行
#   BASE_MODEL_PATH  → Qwen2.5-1.5B-Instruct 基座
#   LORA_PATH        → 古诗微调 LoRA adapter
#   ZIMAGE_PATH      → 本地 Z-Image 绘图 pipeline
# 未配置或路径不存在时，应用会自动隐藏对应的本地选项，仅保留 API 后端
BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH", r"D:\AI_Models\Qwen2.5-1.5B-Instruct")
LORA_PATH       = os.getenv("LORA_PATH",       os.path.join(ROOT_DIR, "models", "poetry_lora"))
ZIMAGE_PATH     = os.getenv("ZIMAGE_PATH",     r"D:\AI_Models\z_image_fp8_full")

# 路径可用性探测（启动期一次性计算，供 UI 过滤下拉项使用）
LOCAL_LLM_AVAILABLE   = os.path.isdir(BASE_MODEL_PATH)
LOCAL_LORA_AVAILABLE  = LOCAL_LLM_AVAILABLE and os.path.isdir(LORA_PATH)
LOCAL_IMAGE_AVAILABLE = os.path.isdir(ZIMAGE_PATH)

# ===== LLM 后端配置 =====
# 默认使用阿里百炼 qwen 系列（与 README 推荐一致）
# 切换 DeepSeek 时改为：LLM_BACKEND="deepseek"、LLM_API_KEY=os.getenv("DEEPSEEK_API_KEY","")、LLM_API_MODEL="deepseek-chat"
LLM_BACKEND   = "qwen"
LLM_API_KEY   = os.getenv("DASHSCOPE_API_KEY", "")
LLM_API_MODEL = "qwen-plus"

# API Key（优先读环境变量）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
ZHIPU_API_KEY     = os.getenv("ZHIPU_API_KEY", "")
MOONSHOT_API_KEY  = os.getenv("MOONSHOT_API_KEY", "")

# ===== 图像后端配置 =====
IMAGE_BACKEND   = "local"
IMAGE_API_MODEL = "qwen-image-max"
# 显存狂暴模式：True=全量权重新进显存（生图秒级，需>12GB显存）
#               False=CPU Offload（省显存但生图慢数倍）
GPU_BEAST_MODE = False

# ===== CLIP 图文一致性评分 =====
# 基于 CLIPScore 论文（Hessel et al., EMNLP 2021）：
#   · 原始余弦相似度在文本-图像生成中自然聚类在 0.15~0.35
#   · ~0.25 为"合理"水平，~0.30 为"明显好"
#   · 中国水墨画（低饱和度）+ 中文诗歌锚点（CLIP 英文训练）额外增加难度
# 参考区间（ViT-B/32 + 中文水墨画）：
#   > 0.30  优秀（该组合的天花板附近）
#   0.25~0.30  良好，图文高度一致
#   0.22~0.25  可接受
#   < 0.22  较差，建议重试
# 升级建议：换 ViT-L/14（约 900 MB）可平均提分 0.02~0.04：
#   CLIP_MODEL_PATH = r"D:\AI_Models\clip-vit-large-patch14"
CLIP_ENABLED     = True
CLIP_THRESHOLD   = 0.22
CLIP_MAX_RETRIES = 2
CLIP_MODEL_PATH = r"D:\AI_Models\clip-vit-base-patch32"
# 诗-提示词语义一致性阈值（CLIP text-text 余弦相似度，低于此值触发提示词重生成）
CLIP_PROMPT_ALIGN_THRESHOLD = 0.15

# ===== CLIP 双锚点权重 =====
CLIP_POEM_WEIGHT   = 0.6   # 诗歌锚点权重（图像与诗歌关键词的匹配度）
CLIP_PROMPT_WEIGHT = 0.4   # 提示词锚点权重（图像与提示词的匹配度）
# 关键词稀疏时（<4 词，哲理/抽象诗常见）降诗锚权重，避免噪声锚点拖低评分
CLIP_SPARSE_POEM_WEIGHT   = 0.3
CLIP_SPARSE_PROMPT_WEIGHT = 0.7
CLIP_SPARSE_WORD_THRESHOLD = 4
# ===== 诗歌生成参数 =====
POEM_CANDIDATE_COUNT = 5
POEM_MAX_TOKENS = 160

# LoRA 模型总上下文窗口（prompt + output），独立于生成上限
LORA_MAX_SEQ_LEN = 320
POEM_TEMPERATURE = 0.8

# ===== 诗歌品质控制 =====
# 废弃线：total 分低于此值的候选直接丢弃，不允许被选为最终结果
# total = (维度加权和 × 重复惩罚) × 必须意象系数，反映综合品质
POEM_QUALITY_THRESHOLD = 0.70
POEM_MAX_DISCARD_PER_BATCH = 2     # 每批 5 首最多允许废弃几首
POEM_MAX_GENERATION_ROUNDS = 3     # 最多几轮生成（含首轮），每轮最多 5 首
POEM_MIN_QUALIFIED = 3             # 至少要有几首合格诗才停止补充生成
POEM_REFINE_TOP_N = 2              # 全自主模式下改几首合格诗

# ===== 提示词生成参数 =====
PROMPT_MAX_TOKENS = 400
PROMPT_TEMPERATURE = 0.9

# ===== 图像生成参数 =====
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
IMAGE_STEPS = 8
IMAGE_GUIDANCE = 0.0
NEGATIVE_PROMPT = ""

# ===== 图像风格映射（中英双语 prompt 后缀）=====
# 单一来源：UI 选项、eval 默认锁定的"水墨画"基线 prompt 都从这里取，
# 避免 eval 路径与生产 UI 路径风格 anchor 漂移。
STYLE_MAP = {
    "水墨画":   "Chinese ink wash painting, sumi-e, monochrome, minimalist, Song Dynasty style",
    "工笔画":   "gongbi fine brushwork, highly detailed, traditional Chinese painting, vivid pigments",
    "写意画":   "xieyi freehand ink painting, expressive spontaneous brushwork, loose poetic strokes",
    "青绿山水": "Chinese blue-green landscape, qinglu style, mineral pigments, Tang Dynasty luminous",
    "油画":     "classical oil painting, rich impasto textures, dramatic chiaroscuro, Renaissance style",
    "卡通插画": "flat vector illustration, clean lines, soft pastel palette, gentle storybook style",
    "浮世绘":   "ukiyo-e woodblock print, bold outlines, flat decorative colors, Edo period Japanese art",
}
STYLE_MAP_CN = {
    "水墨画":   "中国水墨画，素墨写意，极简留白，宋代画风",
    "工笔画":   "工笔细描，精微入微，中国传统工笔，浓丽赋彩",
    "写意画":   "写意水墨，放笔挥洒，诗意笔触，逸气横生",
    "青绿山水": "青绿山水，石青石绿，矿物颜料，唐代金碧辉映",
    "油画":     "古典油画，厚涂肌理，戏剧性明暗对比，文艺复兴风格",
    "卡通插画": "平面矢量插画，线条干净，柔和粉彩，温馨绘本风格",
    "浮世绘":   "浮世绘版画，勾勒分明，平面装饰色彩，江户日本风情",
}


def get_style_suffix(style_name: str, lang: str) -> str:
    """根据语言返回对应风格后缀；未知 style 兜底为水墨画。"""
    if lang == "中文":
        return STYLE_MAP_CN.get(style_name, STYLE_MAP_CN["水墨画"])
    return STYLE_MAP.get(style_name, STYLE_MAP["水墨画"])

# ===== 远程 API 超时与重试 =====
API_TIMEOUT_SUBMIT   = 30    # 任务提交（异步）
API_TIMEOUT_SYNC     = 180   # 同步生图/编辑接口（Z-Image / Qwen-Image / Qwen-Image-Edit）
API_TIMEOUT_POLL     = 15    # 轮询任务状态
API_TIMEOUT_DOWNLOAD = 60    # 图像下载
API_MAX_RETRIES      = 3     # 连接/超时类失败的指数退避重试次数（1s, 2s, 4s）
API_POLL_INTERVAL    = 3     # 轮询间隔
API_POLL_MAX_WAIT    = 120   # 轮询总等待上限

# ===== 评分权重 =====
# WEIGHT_*：完整评分用（scorer.evaluate_full），含 LLM 综合 intent 分
#          → 改诗后的整体评估（每首诗 1 次 LLM 综合调用）
# LOCAL_*_WT（见下文 Arena 配置）：Arena 阶段用（scorer.local_score_poem），仅 topic 一次 LLM
#          → 批量评分（5 首诗只调 1 次 LLM，成本低 5 倍）
# 两套故意分立，不可合并。两套权重总和均为 1.0，但维度不同（intent vs topic）。
WEIGHT_INTENT   = 0.30      # 降权：LLM 主观评分易忽略格律硬伤（如意境好但平仄出律）
WEIGHT_PINGZE   = 0.25      # 升权：平仄是近体诗铁律，应与主观评分平衡
WEIGHT_RHYME    = 0.15
WEIGHT_IMAGERY  = 0.15
WEIGHT_COHESION = 0.15      # 略增：意象逻辑连贯与平仄同等重要
REPETITION_PENALTY_MAX = 0.15

THRESHOLD_PINGZE = 0.8
THRESHOLD_RHYME = 0.8

# 评分聚合下限：penalty/clash/required_coeff 三个惩罚因子叠乘易导致分数溶解
# （例如 0.85*0.75*0.75=0.48），给每个因子加下限避免好诗被多重惩罚吞没。
# 0.7 表示单因子最多扣 30%，三因子叠乘下限 0.343。
SCORE_PENALTY_FLOOR = 0.7

# ===== 堆砌词汇黑名单（硬门控拦截）=====
BAD_PATTERNS = {"我爱", "锦绣", "真美", "辉煌", "璀璨", "旖旎",
                "烂漫", "缱绻", "氤氲", "婆娑", "娉婷",
                "凝眸", "幽梦"}

# ===== Pairwise 锦标赛配置 =====
PAIRWISE_EVOLUTION_ROUNDS = 3   # 每轮产生挑战者与擂主 1v1 对决
CHALLENGERS_PER_ROUND   = 2     # 每轮挑战者数量（不同方向）

# 混合制评分权重（本地客观 + pairwise 审美）
LOCAL_PINGZE_WT   = 0.25   # 平仄
LOCAL_RHYME_WT    = 0.25   # 押韵
LOCAL_IMAGERY_WT  = 0.15   # 意象丰富度
LOCAL_COHESION_WT = 0.15   # 主题连贯性
LOCAL_TOPIC_WT    = 0.20   # 意图契合度（LLM 评分）
# 本地总分权重
ARENA_LOCAL_WT    = 0.75   # arena 阶段本地分占比
ARENA_PAIRWISE_WT = 0.25   # arena 阶段 pairwise 占比
# 进化阶段 pairwise 微调幅度
PAIRWISE_WIN_DELTA  =  0.17  # 挑战者胜的加分
PAIRWISE_LOSE_DELTA = -0.05  # 挑战者败的扣分

# ===== 体裁定义 =====
GENRE_CONFIG = {
    "五言绝句": (4, 5),
    "七言绝句": (4, 7),
    "五言律诗": (8, 5),
    "七言律诗": (8, 7),
}
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
