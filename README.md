# Draft ideas for a new design of the inversion framework

This repo was created as motivation to start implementing some of the ideas
laid out in https://curvenote.com/@simpeg/simpeg-inversion-refactor.

> [!WARNING]
> This repository will never host a stable and well tested codebase. Its
> purpose is to have a place where we can freely try new ideas without having
> to worry about breaking existing code, supporting backward compatibility, or
> providing support to the community.


## Goals

The main goal of the redesign is to create a new public inversion framework in
SimPEG that:

* Is modular, allowing users to plug custom or third-party minimizers,
  simulations, regularizations, etc.
* Can be easily extended (with minimum extra work) to problems outside the
  traditional $\phi(m) = \phi_d(m) + \beta \phi_m(m)$ inversion problem.
* Is implemented in such way that is easier to read, study, and understand by
  beginners that are taking their first steps in inversion theory.
* Is implemented in such way that can be easily extended to more complex and/or
  complicated use cases.
* Defines a clear and minimal interface for each one of the classes.
* Makes use of [abstract classes](https://docs.python.org/3/library/abc.html)
  to enforce implementation of required methods and properties.
* Simplifies the inheritance tree by lowering the amount of inheritance levels.

> [!NOTE]
> These goals are not set in store and are flexible. We are free to add,
> remove, and edit them at any point.

## License

The code in this repository is made available under an *MIT License**.
A copy of this license is provided in [`LICENSE`](LICENSE).
