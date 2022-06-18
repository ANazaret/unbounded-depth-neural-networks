Implementation of the Unbounded Depth Neural Network in PyTorch.

## Quick run
Generates a spiral classification dataset and fit a UDN with fully connected hidden layers.
`python -m experiments.supervised_spiral`

## Model and the Variational Depth
The Unbounded Depth Neural network is implemented in PyTorch as `src.models.UnboundedDepthNetwork`.

The abstract class `src.models.VariationalDepth` represents the variational posterior on the depth L. Any implementation 
of this class can be given to the `UnboundedDepthNetwork`.
- `TruncatedPoisson` implements the variational distribution introduced in the paper.
- `FixedDepth` is a constant distribution simulating  regular (bounded) neural network  

## Training
Some helpful functions for training and evaluating the UDN are available in `src/train.py`.

## Experiments
The three main experiments of the paper (cifar10, spirl, uci) can be reproduced using the code in `experiments`.

## Citation
```

```

 