.. _logging:

Logging and debugging
=====================

The new inversion framework uses the :mod:`logging` module to print messages while inversions are run.
Some classes and functions of the new framework will log events like starting a new iteration, caching a result, updating some parameter, etc.
These events are logged through a global :class:`logging.Logger` that we can access through the :func:`inversion_ideas.utils.get_logger` function.

.. code:: python

   import inversion_ideas as ii

   logger = ii.utils.get_logger()


Logging messages
----------------

Users can use this logger to log messages in their own codes.
For example, we can log an  :obj:`logging.INFO` message through the :meth:`logging.Logger.info` method:

.. code:: python

   import inversion_ideas as ii

   logger = ii.utils.get_logger()
   logger.info("This is a custom message")

.. code::

   [INFO] This is a custom message


Or a :obj:`logging.DEBUG` message through the :meth:`logging.Logger.debug` method:

.. code:: python

   logger.debug("This is a custom debug message")

.. note::

   Debug messages are not shown by default. Learn how to :ref:`adjust the logging level <adjust-logging-level>` so they can be shown.


.. _adjust-logging-level:

Adjusting the logging level
---------------------------

By default this logger will only print out messages of the :obj:`logging.INFO` `level <https://docs.python.org/3/library/logging.html#logging-levels>`__ or above.
Many of the messages logged by the framework's functions and classes are under the :obj:`logging.DEBUG` category.
If we want to receive those messages we need to adjust the logging level of the logger, for example:

.. code:: python

   import inversion_ideas as ii

   logger = ii.utils.get_logger()
   logger.setLevel("DEBUG")

   logger.debug("This is a custom debug message")

.. code::

   [DEBUG] This is a custom debug message




