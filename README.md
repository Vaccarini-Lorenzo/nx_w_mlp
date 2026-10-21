# nx_w_mlp

A PyTorch experiment to play with **multilayer perceptrons (MLPs)** and neural language modeling, based on Bengio et al.'s *A Neural Probabilistic Language Model* (2003).

This is a learning playground, not a full reproduction of the paper.

## Idea

Explore how a neural network can predict the next token from a fixed-size context:

```text
Context tokens -> Learned embeddings -> MLP -> Next-token probabilities
```

The goal is to learn the embeddings and prediction network together, and experiment with how context size, embedding size, and hidden-layer size affect the model.


## Reference

Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and Christian Jauvin. [*A Neural Probabilistic Language Model*](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf). Journal of Machine Learning Research, 3:1137–1155, 2003.
