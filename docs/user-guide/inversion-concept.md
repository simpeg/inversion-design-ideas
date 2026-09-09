# Inversion problem

SimPEG's new inversion framework is targeted to solve **deterministic inversion problems** particularly in the field of geophysics.
These are generally defined as an optimization problem in which we want to minimize a given objective function $\phi(\mathbf{m})$ that depends on a *model* vector {math}`\mathbf{m}` given certain conditions. Mathematically this can be expressed as:

```{math}
\mathbf{m}^* = \min\limits_{\mathbf{m}} \phi(\mathbf{m}),
```

where {math}`\mathbf{m}^*` is the model that minimizes that objective function.

In order to set up this problem we'll need two main pieces: the objective function and the minimizer.

{ref}`objective function <objective-function>`
