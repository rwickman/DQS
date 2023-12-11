import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
my_cmap = mpl.cm.viridis


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

archive_imgs = [
    [
        "/media/data/code/python/DQS/Figures/dqs_archives/walker/max_stag_16_3_archive_ryan_QDWalker2DBulletEnv-v0_species_1.png",
        "/media/data/code/python/pga-map-elites/benchmark_archives/PGA-MAP-Elites/archive_PGA-MAP-Elites_0.5_QDWalker_14_100000.png",
        "/media/data/code/python/pga-map-elites/benchmark_archives/MAP-Elites-ES/archive_MAP-Elites-ES_QDWalker_264030_102000.png",
        "/media/data/code/python/pga-map-elites/benchmark_archives/TD3/archive_TD3-line_QDWalker_12_100078.png",
        "/media/data/code/python/pga-map-elites/benchmark_archives/CMA-MAP-Elites/archive_CMA-ME-line_QDWalker_7_120000.png",
        "/media/data/code/python/pga-map-elites/benchmark_archives/QDPG/archive_QD-RL_QDWalker_17_2_100000.png"
    ],
    [
        "/media/data/code/python/DQS/Figures/dqs_archives/half_cheetah/max_stag_16_4_archive_ryan_QDWalker2DBulletEnv-v0_beh_0.png",
        "/media/data/code/python/pga-map-elites/benchmark_archives/PGA-MAP-Elites/archive_PGA-MAP-Elites_0.5_QDHalfCheetah_47_100000.png",
        "/media/data/code/python/pga-map-elites/benchmark_archives/MAP-Elites-ES/archive_MAP-Elites-ES_QDHalfCheetah_981821_102000.png",
        "/media/data/code/python/pga-map-elites/benchmark_archives/TD3/archive_TD3-line_QDHalfCheetah_16_100136.png",
        "/media/data/code/python/pga-map-elites/benchmark_archives/CMA-MAP-Elites//archive_CMA-ME-line_QDHalfCheetah_60_120000.png",
        "/media/data/code/python/pga-map-elites/benchmark_archives/QDPG/archive_QD-RL_QDHalfCheetah_50_2_100000.png"
    ]
]



fig, axs = plt.subplots(2, 6, gridspec_kw = {'wspace':0, 'hspace':0}, figsize=(20, 7))

labels = ["DQS", "PGA-MAP-Elites", "MAP-Elites-ES", "TD3", "CMA-MAP-Elites", "QD-PG"]
TITLE_FONT_SIZE = 20
LABEL_FONT_SIZE = 18



def get_ax_size(ax):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    # width *= fig.dpi
    # height *= fig.dpi
    return width, height

for i, label in enumerate(labels):
    axs[0][i].set_title(label, fontsize=TITLE_FONT_SIZE)

for i in range(2):
    for j in range(6):
        img = mpimg.imread(archive_imgs[i][j])
        axs[i, j].imshow(img)
        #axs[i, j].axis("off")
        axs[i, j].set_xticklabels([])
        axs[i, j].set_yticklabels([])
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])
        axs[i, j].spines['top'].set_visible(False)
        axs[i, j].spines['right'].set_visible(False)
        axs[i, j].spines['bottom'].set_visible(False)
        axs[i, j].spines['left'].set_visible(False)

    #plt.setp(axs[i, j].spines.values(), visible=False)

axs[0, 0].set_ylabel("QDWalker", fontsize=LABEL_FONT_SIZE)
axs[1, 0].set_ylabel("QDHalfCheetah", fontsize=LABEL_FONT_SIZE)

# Add color bar
# divider = make_axes_locatable(axs[0][-1])

norms = [
    mpl.colors.Normalize(vmin=-17, vmax=2800),
    mpl.colors.Normalize(vmin=-2000, vmax=2782)
]

cax_size = (0.0123, 0.351)
cax_offset = (0.133, 0.01105)

label_pads = [8.9, -5]

# Create the color bars
for i in range(2):
    cax = fig.add_axes([axs[i, -1].get_position().x0 + cax_offset[0], axs[i, -1].get_position().y0 + cax_offset[1], cax_size[0], cax_size[1]])
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norms[i], cmap=my_cmap), cax=cax)

    cbar.set_label("Fitness", size=LABEL_FONT_SIZE, labelpad=label_pads[i])
    cbar.ax.tick_params(labelsize=18)



#plt.subplots_adjust(wspace=0, hspace=0)
# plt.tight_layout()
# print(fig.get_size_inches())
plt.savefig("Figures/archives.pdf", dpi=300, bbox_inches="tight")
# plt.show()