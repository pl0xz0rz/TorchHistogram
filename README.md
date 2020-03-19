# N-dimensional histogramming tool for PyTorch
This repository is an implementation of the histogramdd function from numpy for pytorch Tensors. As a dependency, it uses [torchsearchsorted](https://github.com/aliutkus/torchsearchsorted)

##Description
Implements a function `histogramdd(sample,bins,ranges,weights,device)`
Usage is similar to Numpy, except that:
`sample` is a Tensor with D rows and N columns (matrix transpose of the Numpy version)
`normed` is not supported so far
the output also includes the overflow bins.

##Installation

```
pip install .
```

##Usage

Just import the package. 
```
from torchhistogramdd import histogramdd
```

## Testing

Run the two .py files under teh `test` subfolder