# Generating Music by Langevin Dynamics

We will introduce a new generative model for music composition, applying
Langevin dynamics to a gradient-based score matching algorithm based on [Song
and Ermon, 2019]. Unlike implicit models such as GANs, this learns a true,
explicit distribution of the input data.

![Annealed Langevin dynamics demo](https://i.imgur.com/LXeKezj.jpg)

Previous work has seen a success on modeling from continuous input manifolds,
such high-quality image inpainting and conditional sampling from MNIST,
CIFAR-10, and other datasets. However, it is an open question whether this
algorithm can be adjusted to perform well on discrete domains, such as music
scores.

We hope that Langevin dynamics and score matching can combine the
controllability and of Markov chain Monte Carlo, with the global view and fast
convergence of stochastic gradient descent, to generate high-quality structured,
compositions.

## Problem

[DeepBach] is a simple and controllable autoregressive model for Bach chorale
generation, which are features that make it easy to train and use. In
particular, learning Bach chorales is an interesting task because the music is
highly structured (often following various "rules"), consistent, and often
complex.

![Bach chorale example](https://i.imgur.com/Rf5IH3P.png)

However, there are many instances where DeepBach is unable to capture long-term
structure. Some
[casual listeners](https://www.youtube.com/watch?v=QiBM7-5hA6o&lc=UgimrufXaZHSRHgCoAEC)
have remarked that the compositions "sound good but go nowhere". This could be
due to a combination of reasons: vanishing gradients of the LSTM, and naive
Gibbs sampling procedures being unable to escape 1-optimal local minima.

We believe by applying enough tricks, it should be possible to produce a model
that strongly **avoids these local minima**, while retaining controllability

## Approach

It was seen in [Welling and Teh, 2011] that directing traditional MCMC
algorithms with learned supervision can greatly accelerate their convergence.
This is what motivates us to augment DeepBach's approach with score matching.

It's interesting to analyze other approaches that people have tried in the past:

- **Generative adversarial networks:** Although GANs acheive very promising
  results in modeling latent distributions of images, it's difficult to train
  them on sequence tasks (discrete tokens), as gradients need to propagate from
  the discrminator to the generator ([Yu et al., 2016]).
- **Transformers:** Transformers have been applied to the task of music
  generation and achieved state-of-the-art results on at least one dataset
  ([Huang et al., 2018]). However, transformers are computationally expensive,
  so they're not as controllable through masking and iterative MCMC-like
  algorithms.

We think that score matching and Langevin dynamics, by adding graded noise to
the distribution of data, has the potential to perform well on generative
sequence modeling tasks such as music composition, while maintaining the
_controllability_ of models like DeepBach.

## Evaluation

This project will be successful if we can implement a score matching algorithm
for music generation and evaluate its feasibility. In the best case, score
matching can be used to improve long-term patterns and interpretability.
However, due to the complexity of the algorithm, results are unclear, and we may
need various tricks or innovations to obtian convergence.

Our goal, then, is to determine the tractability and performance of a
score-matching approach in the discrete domain, which we think is very exciting.

[song and ermon, 2019]: https://arxiv.org/abs/1907.05600
[deepbach]: https://arxiv.org/abs/1612.01010
[welling and teh, 2011]:
  https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf
[yu et al., 2016]: https://arxiv.org/abs/1609.05473
[huang et al., 2018]: https://arxiv.org/abs/1809.04281
