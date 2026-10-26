.. _installing:

Installing
==========

Since this framework will not be maintained, you will not find it available in PyPI nor conda-forge. But, it's still possible to install it through pip with:

Latest release
~~~~~~~~~~~~~~

We can specify which release we want to install by pinning the release number by the end of the repository url.

Install `v0.0.1 <https://github.com/simpeg/inversion-design-ideas/releases/tag/v0.0.1>`__ with:

.. code:: bash

   pip install "git+https://github.com/simpeg/inversion-design-ideas@v0.0.1"


Latest development version
~~~~~~~~~~~~~~~~~~~~~~~~~~

If we don't specify any tag, we can install the latest version available in the ``main`` branch:

.. code:: bash

   pip install "git+https://github.com/simpeg/inversion-design-ideas"

.. warning::

   The new inversion framework is still in experimental stages.
   Breaking changes might be introduced without notice.

   We recommend installing a particular release and take note of the version you
   are using, so your code doesn't break if you update your environment.

Install from source
-------------------

Alternatively, we could clone the repository and install the package from source.

1. Clone the repository:

   .. code:: bash

      git clone https://github.com/simpeg/inversion-design-ideas

2. Navigate to the cloned repository:

   .. code:: bash

      cd inversion-design-ideas

3. Use ``pip`` to install it:

   .. code:: bash

      pip install .


