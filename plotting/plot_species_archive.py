import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
from matplotlib.lines import Line2D
import numpy as np

TITLE_FONT_SIZE = 40
LEGEND_FONT_SIZE=30
LEGEND_TITLE_FONT_SIZE=30

walker_archive_img = "/media/data/code/python/DQS/Figures/species_archives/walker/max_stag_16_3_archive_ryan_QDWalker2DBulletEnv-v0_species_1.png"
halfcheetah_archive_img = "/media/data/code/python/DQS/Figures/species_archives/half_cheetah/max_stag_16_4_archive_ryan_QDWalker2DBulletEnv-v0_species_0.png"

fig, axs = plt.subplots(1, 2, figsize=(17, 8))

walker_img = mpimg.imread(walker_archive_img)
halfcheetah_img = mpimg.imread(halfcheetah_archive_img)

axs[0].imshow(walker_img)
axs[1].imshow(halfcheetah_img)

axs[0].set_title("QDWalker", fontsize=TITLE_FONT_SIZE)
axs[1].set_title("QDHalfCheetah", fontsize=TITLE_FONT_SIZE)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# Add the species legend
num_species = 8
norm = mpl.colors.Normalize(vmin=0, vmax=num_species-1)
colormap = ["#ede15b", "#bdcf32", "#87bc45", "#ea5545", "#ef9b20", "#f46a9b", "#b33dc6", "#27aeef", "#edbf33"]

custom_lines = []
labels = []
for i in range(num_species):
    custom_lines.append(Line2D([0], [0], color='w', marker='s', markersize=18, markerfacecolor=colormap[i]))
    species_id = i + 1
    labels.append(f"$z_{species_id}$")


for i in range(2):
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['bottom'].set_visible(False)
    axs[i].spines['left'].set_visible(False)
    axs[i].axis("off")

axs[1].legend(custom_lines, labels, title="Species", loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_TITLE_FONT_SIZE)
plt.tight_layout()
plt.savefig("Figures/species_archives.pdf", dpi=300, transparent=False)

