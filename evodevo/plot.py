import cPickle
import numpy as np
import time
from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from replicators import Individual

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set(color_codes=True, context="poster")
sns.set_style("white", {'font.family': 'serif', 'font.serif': 'Times New Roman'})
colors = sns.color_palette("muted", 3)
sns.set_palette(list(reversed(colors)))

USE_PICKLE = 1
REDUCE_FUNC = np.median
START = time.time()
GENS = 10000
DIR = '/home/sam/Archive/skriegma/rigid_bodies/data'
CMAP = "jet"
GRID_SIZE = 30

if not USE_PICKLE:

    changes = {}

    pickles = glob(DIR+'/Rigid*.p')

    count = 1
    for this_pickle in pickles:
        timer = round((time.time()-START)/60.0, 2)
        print "{0} mins: Getting Run {1}".format(timer, count)
        with open(this_pickle, 'rb') as handle:
            pickle_dict = cPickle.load(handle)

        best = 0
        champ_devo = 1
        for k, v in pickle_dict.items():
            bot = Individual(k, 1)
            bot.weight_matrix = v['weights']
            bot.devo_matrix = v['devo']
            control = bot.calc_control_change()
            body = bot.calc_body_change()
            changes[k] = {'fit': v['fit'], 'control': control, 'body': body}

            if v['fit'] > best:
                best = v['fit']
                champ_devo = body

        print 'best fit: {0}; devo: {1}'.format(best, champ_devo)
        count += 1

    # fit = [x['fit'] for x in changes]
    # control = [x['control'] for x in changes]
    # body = [x['body'] for x in changes]
    #
    # data = [fit, control, body]

    print 'pickling'
    with open(DIR + '/development.p', 'wb') as handle:
        cPickle.dump(changes, handle, protocol=cPickle.HIGHEST_PROTOCOL)

else:
    print 'opening pickle'
    with open(DIR + '/development.p', 'rb') as handle:
        changes = cPickle.load(handle)

fit = []
body = []
control = []
for k, v in changes.items():
    fit += [v['fit']]
    body += [v['body']]
    control += [v['control']]

f, axes = plt.subplots(1, 1, figsize=(6, 5))

print 'plotting'
plt.hexbin(control, body, C=fit,
           gridsize=GRID_SIZE,
           extent=(0, 1, 0, 1),
           cmap=CMAP, linewidths=0.01,
           reduce_C_function=REDUCE_FUNC,
           # vmin=0
           )

axes.set_ylabel("Morphological development", fontsize=15)
axes.set_xlabel("Controller development", fontsize=15)
axes.set_ylim([0, 1])
axes.set_xlim([0, 1])

cb = plt.colorbar()
# cb = plt.colorbar(ticks=range(MIN_BALL_FIT+1, COLOR_LIM+1, 10), boundaries=range(MIN_BALL_FIT, COLOR_LIM+1))
# cb = plt.colorbar(ticks=np.arange(0, 0.25, 0.05))
# cb.set_clim(0, 0.5)
cb.ax.tick_params(labelsize=15)

f.text(0.95, 0.8965, "fitness", fontsize=15, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 0, 'edgecolor': 'white'})
f.text(0.9645, 0.8965 + .042, "High", fontsize=15,  bbox={'facecolor': 'white', 'alpha': 1, 'pad': 0, 'edgecolor': 'white'})
f.text(0.95, 0.1425, "fitness", fontsize=15, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 0, 'edgecolor': 'white'})
f.text(0.971, 0.1425 + 0.042, "Low", fontsize=15, bbox={'facecolor': 'white', 'alpha': 1, 'pad': 0, 'edgecolor': 'white'})

plt.tight_layout()
plt.savefig("Honeycomb.pdf", bbox_inches='tight', transparent=True)
plt.clf()
plt.close()


