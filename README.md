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

##Copyright
Eigenfish is free and open-source software made available under the MIT License. See LICENSE.