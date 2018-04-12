import cPickle
import numpy as np
import time
from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set(color_codes=True, context="poster")
sns.set_style("white", {'font.family': 'serif', 'font.serif': 'Times New Roman'})

USE_PICKLE = 1
START = time.time()
DIR = '/home/sam/Archive/skriegma/rigid_bodies/data'

if not USE_PICKLE:

    fit_dict = {'Evo': {}, 'Devo': {}}

    evo_pickles = glob(DIR+'/Rigid_Evo*.p')
    count = 1
    for this_pickle in evo_pickles:
        timer = round((time.time()-START)/60.0, 2)
        print "{0} mins: Getting Evo Run {1}".format(timer, count)
        with open(this_pickle, 'rb') as handle:
            pickle_dict = cPickle.load(handle)
        for k, v in pickle_dict.items():
            fit_dict['Evo'][k] = v['fit']

        count += 1

    devo_pickles = glob(DIR + '/Rigid_Devo*.p')
    count = 1
    for this_pickle in devo_pickles:
        timer = round((time.time() - START) / 60.0, 2)
        print "{0} mins: Getting Devo Run {1}".format(timer, count)
        with open(this_pickle, 'rb') as handle:
            pickle_dict = cPickle.load(handle)
        for k, v in pickle_dict.items():
            fit_dict['Devo'][k] = v['fit']

        count += 1

    print 'pickling'
    with open(DIR + '/fitness.p', 'wb') as handle:
        cPickle.dump(fit_dict, handle, protocol=cPickle.HIGHEST_PROTOCOL)

else:
    print 'opening pickle'
    with open(DIR + '/fitness.p', 'rb') as handle:
        fit_dict = cPickle.load(handle)

evo_id = []
evo = []
for k, v in fit_dict['Evo'].items():
    evo_id += [k]
    evo += [1000*v]

devo_id = []
devo = []
for k, v in fit_dict['Devo'].items():
    devo_id += [k]
    devo += [1000*v]


f, axes = plt.subplots(1, 1, figsize=(6, 5))

print 'plotting'
# plt.plot(evo_id, evo)
plt.plot(devo_id, devo)

plt.tight_layout()
plt.savefig("Fitness.png", bbox_inches='tight')

