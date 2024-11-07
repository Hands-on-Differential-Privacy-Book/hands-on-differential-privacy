import matplotlib.pyplot as plt
import numpy as np

# set up the figure
fig = plt.figure()
ax = fig.add_subplot(111)
fig.set_figwidth(10)
fig.set_figheight(3)

# draw lines
xmin = -10
xmax = 10
y = 5
height = 1
linewidth = .75

ax.set_xlim(xmin - 1, xmax + 1)
ax.set_ylim(3, 6)

plt.hlines(y, xmin, xmax, color='black', linewidth=1)

def tick(value, height):
    plt.vlines(value, y - height / 2., y + height / 2., color='black', linewidth=.5)

def label(value, height, text=None):
    plt.text(value, y + height / 2 + .1, text or value, horizontalalignment='center')

def bin(x1, x2):
    plt.hlines(y - height, x1 + .1, x2 - .1, color='black', linewidth=1)

def bin_pt(x):
    plt.scatter(x, y - height, color='black', s=1)

def bin_label(x, text):
    plt.text(x, y - height - .5, text, 
             horizontalalignment='center', rotation = -30,
             rotation_mode='anchor')


bands = 1, 2, 4, 8

for band in bands:
    # negative bands
    tick(-band, 1)
    label(-band, 1)

    # positive bands
    tick(band, 1)
    label(band, 1)

    # mantissa values
    for value in np.linspace(band, band / 2, num=5):
        tick(-value, .5)
        tick(value, .5)

    # bins
    idx = np.log2(band).astype(int) + 1074
    edges = -band, -band / 2
    bin(*edges)
    bin_label(np.mean(edges), -idx)

    edges = band / 2, band
    bin(*edges)
    bin_label(np.mean(edges), idx)

# # subnormals
# for value in np.linspace(0, bands[0] / 2, num=5):
#     tick(-value, .5)
#     tick(value, .5)

# 0
tick(0, 1)
label(0, 1)
bin_pt(0)
bin_label(0, 0)

# -infty
tick(xmin, 1)
label(xmin, 1, '-∞')
bin_pt(xmin)
bin_label(xmin, -2099)

# infty
tick(xmax, 1)
label(xmax, 1, '∞')
bin_pt(xmax)
bin_label(xmax, 2099)

d = .075
o = .075
def cut(v):
    ax.plot((v - (o + d) / 2, v + (o + d) / 2), (y, y), color='white')
    ax.plot((v - o - d, v + o - d), (y - o, y + o), color='black')
    ax.plot((v - o + d, v + o + d), (y - o, y + o), color='black')

    ax.plot((v - o - d, v + o - d), (y - o - height, y + o - height), color='black')
    ax.plot((v - o + d, v + o + d), (y - o - height, y + o - height), color='black')

cuts = np.mean([xmin, -bands[-1]]), np.mean([xmax, bands[-1]])
for value in cuts:
    cut(value)

cut(-.25)
cut(.25)

plt.axis('off')
plt.savefig("../images/ch06_private_bounds_bands.png", bbox_inches='tight')
plt.show()