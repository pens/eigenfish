# Eigenfish
Eigenfish is a Python 3 package for detecting fish in an images.
*Requires python3, scipy, numpy and scikit-learn. sphinx also required to 
build documentation.*

## Usage
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
See `process/process.py` and `classify/classify.py` for the defaults.

## Documentation
Documentation is available under `eigenfish/doc/_build/html/index.html.`

The documentation generator is available in `doc/`. Documentation can be rebuilt
by running `make html` from `doc/`.

## Unit tests
Unit tests are in the `test/` subdirectory. To run the tests, call
`python3 -m unittest test/test.py`.

## Example
The example script shows the general use of the Eigenfish package. The package
can be run with `python3 example/example.py`.

## Copyright
Eigenfish is free and open-source software made available under the MIT License.
See [LICENSE file](LICENSE) for details. 

## References

[1] Candès, E. J., Li, X., Ma, Y., and Wright, J. Robust principal component analysis? Journal of the ACM, 58(3):11:1-11:37, 2011.<br />
[2] Huang, P.X., Boom, B.J., and Fisher, R.B., Underwater live fish recognition using a balance-guaranteed optimized tree, In Computer Vision ACCV 2012, Lecture Notes in Computer Science Volume 7724, pp. 422- 433, Springer Berlin Heidelberg, 2013.
