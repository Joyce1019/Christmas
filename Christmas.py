import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import random
from PIL import Image
from matplotlib.font_manager import FontProperties
import os
# ====================== 配置部分 ======================
np.random.seed(2025)
random.seed(2025)
current_style_id = 5  # 1-5

styles = {
    "1": {
        "bg_col": "#0f0505",
        "tree_cols": ["#0B3D0B", "#144514", "#006400"],
        "trunk_col": "#3e2723",
        "decor_cols": ["#FFD700", "#CCA43B", "#8B0000"],
        "star_col": "#FFD700",
        "text_col": "#FFD700",
        "snow_col": "white",
        "ribbon": True,
        "ribbon_col": "#D4AF37",
        "ribbon_width": 1.5
    },
    "2": {
        "bg_col": "#0f0505",
        "tree_cols": ["#4A90E2", "#6BB0F5", "#8CC6F8"],
        "trunk_col": "#8B6E46",
        "decor_cols": ["#FFFFFF", "#E0F7FA", "#B3E5FC"],
        "star_col": "#FFFFFF",
        "text_col": "#265A88",
        "snow_col": "#B3E5FC",
        "ribbon": True,
        "ribbon_col": "#FFFFFF",
        "ribbon_width": 1.2
    },
    "3": {
        "bg_col": "#0f0505",
        "tree_cols": ["#8B5A2B", "#A67C52", "#C19A6B"],
        "trunk_col": "#5D4037",
        "decor_cols": ["#FFD700", "#FFC107", "#FFB300"],
        "star_col": "#FFD700",
        "text_col": "#FFE0B2",
        "snow_col": "#FFFFFF",
        "ribbon": True,
        "ribbon_col": "#FFD700",
        "ribbon_width": 1.3
    },
    "4": {
        "bg_col": "#0A0A0A",
        "tree_cols": ["#00D1FF", "#00B8E6", "#009FC7"],
        "trunk_col": "#1A1A1A",
        "decor_cols": ["#FF6600", "#0099FF", "#FF00CC"],
        "star_col": "#FF6600",
        "text_col": "#0099FF",
        "snow_col": "#1A1A22",
        "ribbon": True,
        "ribbon_col": "#0099FF",
        "ribbon_width": 1.0
    },
    "5": {
        "bg_col": "#0f0505",
        "tree_cols": ["#F8BBD0", "#F48FB1", "#EC407A"],
        "trunk_col": "#D7CCC8",
        "decor_cols": ["#FFFFFF", "#FFF8E1", "#FFD700"],
        "star_col": "#FFD700",
        "text_col": "#AD1457",
        "snow_col": "#FCE4EC",
        "ribbon": True,
        "ribbon_col": "#FFFFFF",
        "ribbon_width": 0.6
    }
}
cfg = styles[str(current_style_id)]


# ====================== 字体容错 ======================
def get_windows_art_font():
    try:
        from matplotlib.font_manager import FontManager
        fm = FontManager()
        available = {f.name for f in fm.ttflist}
    except Exception:
        return "Arial"

    candidates = ["STXingkai", "STHupo", "FZShuTi", "STCaiyun", "STKaiti", "Arial"]
    for f in candidates:
        if f in available:
            return f
    return "Arial"


# ====================== 核心函数 ======================
def get_star_polygon(x_center=0, y_center=0, radius=0.1):
    angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    angles += np.pi / 2
    radii = np.array([radius, radius * 0.38] * 5)
    x = x_center + radii * np.cos(angles)
    y = y_center + radii * np.sin(angles)
    return np.column_stack((x, y))


def generate_tree_data(cfg):
    # 树叶
    n_leaves = 4000
    h = np.random.uniform(0, 1, n_leaves)
    base_r = (1 - h)
    layer_cycle = (h * 7) % 1
    r = base_r * 0.65 * (0.4 + 0.6 * (1 - layer_cycle) ** 0.7)
    theta = np.random.uniform(0, 2 * np.pi, n_leaves)

    df_tree = pd.DataFrame({
        "x": r * np.cos(theta),
        "y": h - 0.5,
        "z": r * np.sin(theta),
        "col": np.random.choice(cfg["tree_cols"], n_leaves),
        "size": np.random.uniform(0.6, 1.8, n_leaves),
        "type": "tree",
        "alpha": 0.95
    })

    # 树干
    n_trunk = 500
    h_trunk = np.random.uniform(-0.7, -0.45, n_trunk)
    r_trunk = 0.12
    theta_trunk = np.random.uniform(0, 2 * np.pi, n_trunk)

    df_trunk = pd.DataFrame({
        "x": r_trunk * np.cos(theta_trunk),
        "y": h_trunk,
        "z": r_trunk * np.sin(theta_trunk),
        "col": cfg["trunk_col"],
        "size": 1.2,
        "type": "trunk",
        "alpha": 1.0
    })

    # 装饰
    n_decor = 300
    h_dec = np.random.uniform(0, 0.95, n_decor)
    base_r_dec = (1 - h_dec)
    layer_cycle_dec = (h_dec * 7) % 1
    r_dec = base_r_dec * 0.68 * (0.4 + 0.6 * (1 - layer_cycle_dec) ** 0.7)
    theta_dec = np.random.uniform(0, 2 * np.pi, n_decor)

    df_decor = pd.DataFrame({
        "x": r_dec * np.cos(theta_dec),
        "y": h_dec - 0.5,
        "z": r_dec * np.sin(theta_dec),
        "col": np.random.choice(cfg["decor_cols"], n_decor),
        "size": np.random.uniform(2, 4, n_decor),
        "type": "decor",
        "alpha": 1.0
    })

    frames = [df_trunk, df_tree, df_decor]

    # 彩带
    if cfg.get("ribbon", False) and cfg.get("ribbon_col", None) is not None:
        n_rib = 3000
        h_rib = np.linspace(0, 0.95, n_rib)
        base_r_rib = (1 - h_rib) * 0.65 * 1.05
        theta_rib = 10 * np.pi * h_rib
        df_ribbon = pd.DataFrame({
            "x": base_r_rib * np.cos(theta_rib),
            "y": h_rib - 0.5,
            "z": base_r_rib * np.sin(theta_rib),
            "col": cfg["ribbon_col"],
            "size": cfg["ribbon_width"],
            "type": "ribbon",
            "alpha": 1.0
        })
        frames.append(df_ribbon)

    return pd.concat(frames, ignore_index=True)


def generate_snow(cfg, n_flakes=150):
    return pd.DataFrame({
        "x": np.random.uniform(-1, 1, n_flakes),
        "y": np.random.uniform(-0.8, 1.2, n_flakes),
        "z": np.random.uniform(-1, 1, n_flakes),
        "col": cfg["snow_col"],
        "size": np.random.uniform(0.5, 2, n_flakes),
        "type": "snow",
        "alpha": np.random.uniform(0.5, 0.9, n_flakes),
        "speed": np.random.uniform(0.015, 0.035, n_flakes)
    })


def process_frame(frame_id, static_data, snow_data, n_frames):
    angle = 2 * np.pi * (frame_id / n_frames)

    tree_rot = static_data.copy()
    tree_rot["x_rot"] = tree_rot["x"] * np.cos(angle) - tree_rot["z"] * np.sin(angle)
    tree_rot["z_rot"] = tree_rot["z"] * np.cos(angle) + tree_rot["x"] * np.sin(angle)
    tree_rot["y_final"] = tree_rot["y"]

    snow_curr = snow_data.copy()
    snow_curr["y_final"] = -0.8 + (snow_curr["y"] - frame_id * snow_curr["speed"] - (-0.8)) % 2
    snow_curr["x_rot"] = snow_curr["x"]
    snow_curr["z_rot"] = snow_curr["z"]

    all_data = pd.concat([tree_rot, snow_curr], ignore_index=True)

    # 透视投影
    all_data["depth"] = 1 / (2.5 - all_data["z_rot"])
    all_data["x_proj"] = all_data["x_rot"] * all_data["depth"] * 2
    all_data["y_proj"] = all_data["y_final"] * all_data["depth"] * 2
    all_data["size_vis"] = all_data["size"] * all_data["depth"] * 1.5

    # 透明度
    all_data["alpha_vis"] = all_data["alpha"]
    tree_mask = all_data["type"] != "snow"
    all_data.loc[tree_mask, "alpha_vis"] = (
        all_data.loc[tree_mask, "alpha"] * (all_data.loc[tree_mask, "z_rot"] + 1.2) / 2.2
    )
    all_data["alpha_vis"] = np.clip(all_data["alpha_vis"], 0.08, 1.0)

    return all_data.sort_values("depth")


# ====================== 数据 ======================
static_data = generate_tree_data(cfg)
snow_data = generate_snow(cfg, 150)

# ====================== 画布 ======================
fig, ax = plt.subplots(figsize=(8, 10), dpi=80)
fig.patch.set_facecolor(cfg["bg_col"])
ax.set_facecolor(cfg["bg_col"])
ax.set_xlim(-0.8, 0.8)
ax.set_ylim(-0.8, 0.9)
ax.set_aspect("equal")
ax.axis("off")

# 星星
star_coords = get_star_polygon(x_center=0, y_center=0.43, radius=0.03)
star_patch = mpatches.Polygon(
    star_coords,
    facecolor=cfg["star_col"],
    edgecolor="white",
    linewidth=1.0,
    zorder=1000,
    alpha=1.0
)
ax.add_patch(star_patch)

# 文字
font_prop = FontProperties(family="Comic Sans MS")  # 或 Brush Script MT
text = ax.text(
    0, 0.65, "Merry Christmas\n I love you",
    fontproperties=font_prop,
    fontweight="bold",
    color=cfg["text_col"],
    fontsize=32,
    ha="center", va="center",
    zorder=999,
    style="italic"
)

# scatter 初始化
scatter = ax.scatter([0], [0], s=[10], c=["white"], marker="o", zorder=100)

# ====================== 载入并显示图片 ======================
img_path = r"C:\Users\27391\OneDrive\Desktop\3.jpg"

img = Image.open(img_path).convert("RGB")

# 图片原始宽高比
w, h = img.size
aspect = h / w   # 高 / 宽

# ===== 控制图片“宽度”即可，其它自动算 =====
img_width = 0.42          # 在你的坐标系里的宽度（可微调：0.3~0.4）
img_height = img_width * aspect

# ===== 居中 & 放在画布底部 =====
x_center = 0.0
y_bottom = -0.75          # 靠近底部（你的 ylim 是 -0.8）

extent = (
    x_center - img_width / 2,
    x_center + img_width / 2,
    y_bottom,
    y_bottom + img_height
)

img_artist = ax.imshow(
    img,
    extent=extent,
    zorder=150,            # 在树前但不挡文字
    alpha=1.0)

# ====================== 动画 ======================
n_frames = 60
fps = 15

def draw_frame(frame):
    fd = process_frame(frame, static_data, snow_data, n_frames)
    scatter.set_offsets(fd[["x_proj", "y_proj"]].values)

    rgba = mcolors.to_rgba_array(fd["col"].values)
    rgba[:, 3] = fd["alpha_vis"].values
    scatter.set_facecolors(rgba)

    scatter.set_sizes(fd["size_vis"].values)

def init():
    draw_frame(0)
    return scatter, star_patch, text

def update(frame):
    draw_frame(frame)
    return scatter, star_patch, text

# ✅ 保存 gif 最稳：blit=False
ani = animation.FuncAnimation(
    fig, update,
    frames=n_frames,
    init_func=init,
    interval=1000 / fps,
    blit=False,
    repeat=True
)

print("正在生成圣诞动画...")
out_gif = "christmas_tree_animation.gif"
ani.save(
    out_gif,
    writer="pillow",
    fps=fps,
    dpi=80,
    savefig_kwargs={"facecolor": cfg["bg_col"]}
)
print("动画生成完成！文件保存为：", out_gif)

# ====================== 验证GIF帧数（关键排查） ======================
im = Image.open(out_gif)
n = 0
try:
    while True:
        im.seek(n)
        n += 1
except EOFError:
    pass

with open("debug_nframes.txt", "w", encoding="utf-8") as f:
    f.write(f"GIF frames = {n}\n")

print("GIF帧数 =", n, "（已写入 debug_nframes.txt）")

plt.show()