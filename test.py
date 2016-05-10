from eigenfish import Eigenfish
import numpy
import scipy.misc

img = "D:/Data/Research Data/Eigenfish/160313/Config_2/camera1_image_%d.tif"

im_se1 = list(range(280, 300))
im_se2 = list(range(20, 40))

shape = scipy.misc.imread(img % im_se1[0], True).shape

fish = numpy.empty((shape[0] * shape[1], len(im_se1)))
for i in range(len(im_se1)):
    fish[:, i] = scipy.misc.imread(img % im_se1[i], True).flatten()
no_fish = numpy.empty((shape[0] * shape[1], len(im_se2)))
for i in range(len(im_se2)):
    no_fish[:, i] = scipy.misc.imread(img % im_se2[i], True).flatten()

ef = Eigenfish(shape)

fi = round(.8 * len(im_se1))
ni = round(.8 * len(im_se2))

train = numpy.hstack((fish[:, 0:fi], no_fish[:, 0:ni]))
labels_train = ["fish" for i in range(fi)] + ["no fish" for i in range(ni)]
cross = numpy.hstack((fish[:, fi:], no_fish[:, ni:]))
labels_cross = (["fish" for i in range(fi, len(im_se1))] +
                ["no fish" for i in range(ni, len(im_se2))])

ef.train(train, labels_train)

print("Percent correct: " + str(100 * ef.cross_validate(cross, labels_cross)))
