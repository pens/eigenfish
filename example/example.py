import os
import shutil
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import eigenfish
import util

if __name__ == "__main__":
    print("Loading training images")
    fish_imgs = (["example/images/fish/%d.jpg" % i for i in range(0, 15)] +
                 ["example/images/nofish/%d.jpg" % i for i in range(0, 15)])
    fish_mat, shape = util.load_img_mat(fish_imgs)

    print("Training model")
    ef = eigenfish.Eigenfish(shape)
    ef.train(fish_mat, (["fish" for i in range(0, 15)] +
                        ["no fish" for i in range(0, 15)]))

    print("Loading test images")
    test_imgs = (["example/images/fish/%d.jpg" % i for i in range(15, 20)] +
                 ["example/images/nofish/%d.jpg" % i for i in range(15, 20)])
    test_mat = util.load_img_mat(test_imgs)[0]

    print("Classifying test images")
    labels = ef.classify(test_mat)
    print("Labels:")
    print(labels)

    print("Cross-validating test images")
    pct = ef.cross_validate(test_mat, (["fish" for i in range(5)] +
                                       ["no fish" for i in range(5)]))
    print("Percent correct:")
    print(str(pct * 100) + "%")

    print("Saving trained model")
    ef.save("model/example.ef")

    print("Reloading saved model")
    del ef
    ef = eigenfish.Eigenfish(shape)
    ef.load("model/example.ef")
    shutil.rmtree("model/", True)
