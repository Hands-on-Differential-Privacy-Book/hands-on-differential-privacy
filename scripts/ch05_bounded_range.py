import matplotlib.pyplot as plt

# set up the figure
fig = plt.figure()
ax = fig.add_subplot(111)
fig.set_figwidth(8)
fig.set_figheight(2)

# draw lines
y = 5

epsilon = 1.
t = epsilon * .757

xmin = -epsilon * 1.5
xmax = -xmin


plt.hlines(y, xmin, xmax, color='black', linewidth=1)

def tick(value, height):
    plt.vlines(value, y - height / 2., y + height / 2., color='black', linewidth=.5)

def label(value, height, text=None):
    plt.text(value, y + height / 2 + .1, text or value, horizontalalignment='center')


tick(-epsilon, 1)
label(-epsilon, 1, "$-\epsilon$")

tick(t - epsilon, .5)
label(t - epsilon, .5, "$t - \epsilon$")

tick(0, .1)
label(0, .1, "$0$")

tick(t, .5)
label(t, .5, "$t$")

tick(epsilon, 1)
label(epsilon, 1, "$\epsilon$")

plt.hlines(y - .25, t - epsilon, t, color='black', linewidth=1)

plt.axis('off')
plt.savefig("../images/ch05_bounded_range.png", bbox_inches='tight')
plt.show()
