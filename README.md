# Eigenfish
Eigenfish is a Python 3 package for detecting fish in an image sequence.

## Requirements
Requires *Python 3*.
To install, run `pip3 install -r requirements.txt` from the root directory.

## Usage
*For a detailed functional example, please see [example.py](example.py).*
*Documentation is available at [docs/_build/html/index.html](docs/_build/html/index.html).*

Eigenfish must be trained before it is able to classify an images as follows:
```
import eigenfish

ef = eigenfish.Eigenfish(image_shape)
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

`util.py` contains the helper function load_img_mat to make loading images easier.

### Save/Load
Eigenfish has support for saving a trained classifier and loading it later,
through `Eigenfish.save(filename)` and `Eigenfish.load(filename)`.

### Customization
Custom classifiers and preprocessors can be used with Eigenfish by passing
classes to the `processor` and `classifier` arguments in the constructor
`Eigenfish()`.
See `process/process.py` and `classify/classify.py` for the defaults.

### Documentation
Run `make html` from the `docs/` directory.

### Example
Run `python3 example.py` from the root directory.

### Unit Tests
Run `python3 test.py` from the root directory.

## Copyright
Eigenfish is free and open-source software made available under the MIT License.
See [LICENSE file](LICENSE) for details.

## References
[1] Candès, E. J., Li, X., Ma, Y., and Wright, J. Robust principal component analysis? Journal of the ACM, 58(3):11:1-11:37, 2011.

[2] Huang, P.X., Boom, B.J., and Fisher, R.B., Underwater live fish recognition using a balance-guaranteed optimized tree, In Computer Vision ACCV 2012, Lecture Notes in Computer Science Volume 7724, pp. 422- 433, Springer Berlin Heidelberg, 2013.
