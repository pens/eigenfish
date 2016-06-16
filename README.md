# Eigenfish
Eigenfish is a Python package for detecting fish in an images.

##TODO
- Multiclass ML
- GPU Support

##Usage
*A full example script is available at example/example.py.*

Eigenfish is used as follows:
```
ef = Eigenfish(image_shape)
ef.train(image_matrix, labels)
result = ef.classify(unlabeled_image_matrix)
```
where:
- `image_shape` is the `(height, width)` of all images used
- `image_matrix` is matrix with each column a flattened image
- `labels` is a list of labels with `labels[i]` corresponding to
`image_matrix[:, i]`
- `unlabeled_image_matrix` is the matrix of flattened images to classify

Additionally, `ef.cross_validate(labeled_image_matrix, labels)` can be called
after `ef.train(...)` to check accuracy of the trained model.

Eigenfish has support for saving a trained classifier and loading it later,
through `Eigenfish.save(filename)` and `Eigenfish.load(filename)`.

Custom classifiers and preprocessors can be used with Eigenfish by passing
classes to the `processor` and `classifier` arguments in the constructor
`Eigenfish()`.
See process/process.py and classify/classify.py for the defaults.

##Documentation
Documentation is available under eigenfish/doc/_build/html/index.html.

Documentation can be rebuilt by calling `make html` from the root directory.

##Copyright
Eigenfish is free and open-source software made available under the MIT License.
See LICENSE.