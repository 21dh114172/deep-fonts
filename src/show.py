import h5py, random, numpy
import PIL, PIL.Image
from scipy.ndimage import filters
import random
import os

from dotenv import load_dotenv
load_dotenv()


f = h5py.File(os.getenv("DATA_PATH") + '/fonts.hdf5', 'r')
data = f['fonts']
print data.shape

i = random.randint(0, data.shape[0]-1)
for z in xrange(10):
    j = random.randint(0, data.shape[1]-1)
    m = data[i][j]
    m = filters.gaussian_filter(m, sigma=random.random()*1.0)
    img = PIL.Image.fromarray(numpy.uint8(255 - m))
    img.show()



