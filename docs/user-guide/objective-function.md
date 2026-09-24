---
file_format: mystnb
kernelspec:
  display_name: Python 3
  name: python3
---

# Objective functions

Any objective function is defined as a vector function {math}`\phi: \mathbb{R}^M \rightarrow \mathbb{R}`, i.e. it takes a model vector {math}`\mathbf{m}` with {math}`M` elements and returns a single value.

In the new SimPEG's inversion framework, objective functions are represented by a child of the {py:class}`inversion_ideas.base.Objective`, like the {py:class}`inversion_ideas.DataMisfit` for example.

```{code-cell} python
import numpy as np
import inversion_ideas as ii

# Define a linear regressor as a simulation
n_data, n_params = 10, 15
simulation = ii.simulations.LinearRegressor.create_random(
    n_data, n_params, build_hessian=True, seed=42
)

# Create some random data and uncertainties
data = np.random.default_rng(seed=1414).uniform(size=n_data)
uncertainties = 1e-3 * np.ones_like(data)

# Define a data misfit function
data_misfit = ii.DataMisfit(data, uncertainties, simulation)
data_misfit
```

Once defined, the objective functions can only be evaluated on model vectors with ``n_params`` elements. We can check that property on every objective function:

```{code-cell} python
print("Expected size of model vector:", data_misfit.n_params)
```

Objective functions can be evaluated by calling them:

```{code-cell} python
# Define some model
model = np.arange(n_params, dtype=np.float64)

# Evaluate the data_misfit on that model
data_misfit(model)
```

All objective functions have two methods to compute its derivatives:

* {py:meth}`~inversion_ideas.base.Objective.gradient` computes the gradient of the objective function: {math}`\nabla \phi(\mathbf{m})`.
* {py:meth}`~inversion_ideas.base.Objective.hessian` computes the Hessian of the objective function, i.e. the matrix of the second derivatives: {math}`\nabla^2 \phi(\mathbf{m})`.

```{code-cell} python
# Get gradient
gradient = data_misfit.gradient(model)
gradient
```

```{code-cell} python
# Get Hessian
hessian = data_misfit.hessian(model)
hessian
```

> By default, the {py:class}`~inversion_ideas.DataMisfit` and many other
> objective functions in the framework won't build the whole Hessian matrix, but
> return a {py:class}`~scipy.sparse.linalg.LinearOperator` instead.
>
> Hessian matrices are ``n_params`` {math}`\times` ``n_params`` square matrices
> that can grow very large when ``n_params`` is big.
> A {py:class}`~scipy.sparse.linalg.LinearOperator` allows us to operate with such
> matrix as if it were an array, but without storing it entirely in memory.


## Renaming a function

An objective function can be renamed using the {py:meth}`~inversion_ideas.base.Objective.set_name` method. For example:

```{code-cell} python
data_misfit.set_name("data_misfit")
data_misfit
```

Let's rename it to its previous name:

```{code-cell} python
data_misfit.set_name("d")
data_misfit
```

> The {py:meth}`~inversion_ideas.base.Objective.set_name` method returns the same objective function, so we can use it directly when defining a new one. For example:
>
> ```python
> data_misfit = ii.DataMisfit(data, uncertainties, simulation).set_name("data_misfit")
> ```

## Combining objective functions

### Add two objective functions

It's possible to define any linear combination of objective functions.

Consider we have a {py:class}`~inversion_ideas.TikhonovZero` regularization function:

```{code-cell} python
zero = ii.TikhonovZero(n_params)
zero
```

We can add it to the ``data_misfit`` one by summing them together to obtain a new objective function:

```{code-cell} python
phi = data_misfit + zero
phi
```

This {py:class}`inversion_ideas.base.Combo` is an objective function that works as its mathematical counterpart {math}`\phi(\mathbf{m}) = \phi_\text{d}(\mathbf{m}) + \phi_\text{0}(\mathbf{m})` does: we can evaluate it on a given model, get its gradient and Hessian:

```{code-cell} python
phi(model)
```

```{code-cell} python
phi.gradient(model)
```

```{code-cell} python
phi.hessian(model)
```

> The {py:class}`inversion_ideas.base.Combo` class is not meant to be
> instantiated. Add two or more objective functions together to obtain one.

The {py:class}`inversion_ideas.base.Combo` works as a collection of objective
functions. This means that we can get the number of objective functions it
contains through the ``len`` function, and we can also access each one of its
terms by index:

```{code-cell} python
len(phi)
```

```{code-cell} python
print(phi[0])
print(phi[1])
```

```{code-cell} python
for i, term in enumerate(phi):
    print(f"Term {i}: {term}")
```

```{code-cell} python
print(data_misfit in phi)
print(zero in phi)
```

### Add multiple objective functions

We can also add more than two objective functions together. For example, consider that we want to add the ``data_misfit`` and the ``zero`` regularization with a {py:class}`~inversion_ideas.TikhonovFirst` regularization:

> **TODO:** Use TikhonovFirst here!

```{code-cell} python
first = ii.TikhonovZero(n_params).set_name("1")
first
```

```{code-cell} python
phi = data_misfit + zero + first
phi
```

By default, when adding more than two objective functions together, we'll get
a *nested* {py:class}`inversion_ideas.base.Combo` object: the first element
will be a {py:class}`inversion_ideas.base.Combo` containing the `data_misfit`
and the `zero` regularization, and the second element will be the `first`
regularization.

```{code-cell} python
print(len(phi))
print(phi[0])
print(phi[1])
```

And since the first element is a {py:class}`~inversion_ideas.base.Combo`, we can also treat it as a collection:

```{code-cell} python
print(len(phi[0]))
print(phi[0][0])
print(phi[0][1])
```

If we don't want to have this kind of structure,  we can
*flatten* any {py:class}`~inversion_ideas.base.Combo` through the
{py:meth}`~inversion_ideas.base.Combo.flatten` method:

```{code-cell} python
flat_phi = phi.flatten()
flat_phi
```

```{code-cell} python
len(flat_phi)
```

```{code-cell} python
print(flat_phi[0])
print(flat_phi[1])
print(flat_phi[2])
```

```{code-cell} python
data_misfit in flat_phi
```
```{code-cell} python
zero in flat_phi
```
```{code-cell} python
first in flat_phi
```

#### Is in or contains?

When using the ``in`` statement to check whether an objective function is part of a
{py:class}`~inversion_ideas.base.Combo`, we are only checking if that function is one of its elements, but not a recursive search through its nested
{py:class}`~inversion_ideas.base.Combo`.

For example, in we know that `first` is the second element of `phi`:

```{code-cell} python
phi
```

```{code-cell} python
phi[1]
```

So, we know that `first in phi` should be `True`:


```{code-cell} python
first in phi
```

But, the first element of `phi` is a {py:class}`~inversion_ideas.base.Combo`
containing `data_misfit` and `zero`, so `data_misfit in phi` should be `False`:

```{code-cell} python
data_misfit in phi
```

And the same goes for zero:

```{code-cell} python
zero in phi
```

Alternatively, we can use the {py:meth}`~inversion_ideas.base.Combo.contains`
method to check recursively for a particular term:

```{code-cell} python
phi.contains(data_misfit)
```

```{code-cell} python
phi.contains(zero)
```

```{code-cell} python
phi.contains(first)
```

