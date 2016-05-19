#TODO
- Enable special members (e.g. `__init__`) in Sphinx documentation
- Add copyright headers
- Improve performance of CPU RPCA
- Add GPU code

# Eigenfish
Python package for detecting fish in an image sequence.

##Usage
Eigenfish is used as follows:
```
ef = Eigenfish(image_shape)
ef.train(labeled_image_matrix, labels)
result = ef.classify(unlabeled_image_matrix)
```
Additionally, `ef.cross_validate(labeled_image_matrix, labels)` can be called
after `ef.train(...)` on a different set of labelled images to check accuracy.

Eigenfish has support for saving a trained classifier and loading it later,
through `Eigenfish.save(filename)` and `Eigenfish.load(filename)`.

Custom classifiers and preprocessors can be used with Eigenfish by passing
classes to the `processor` and `classifier` arguments in `Eigenfish.__init__()`.
See process/process.py and classify/classify.py for the default classes.

##Documentation
Documentation is available under eigenfish/doc/_build/html/index.html.

Documentation can be rebuilt by calling `make html` from the root directory.

##Copyright
Eigenfish is free and open-source software made available under the MIT License.
See LICENSE.