import random
import numpy
import random
import model

data = model.get_data()
n, k = data.shape[0], data.shape[1]

m = model.Model(n, k)
m.try_load()
run_fn = m.get_run_fn()

train_set, test_set = m.sets()
chars = {}
for i, j in test_set:
    chars.setdefault(j, []).append(i)

batch_is = numpy.zeros((k,), dtype=numpy.int32)
batch_js = numpy.zeros((k,), dtype=numpy.int32)
for z in xrange(k): # k is 62, length of A-Za-z0-9
    batch_is[z] = random.choice(chars[z]) # choose random font from test set
    batch_js[z] = z # from A-Z a-z 0-9, a is 26

batch_pred = run_fn(batch_is, batch_js)
combined = numpy.zeros((2*k, 64 * 64))
for z in xrange(k):
    combined[2*z] = data[batch_is[z]][z].flatten() * 1.0 / 255 # real font
    combined[2*z+1] = batch_pred[z] # predict font
model.draw_grid(combined).save('real_vs_pred2.png')
