
Single-shot Hankel transforms
=============================

The functions below are more convenient version of the Hankel transform
functions, designed for when you want to transform only a single function.
They basically create a throw-away :class:`.HankelTransform` object, transform the input
onto the appropriate grid using :meth:`.HankelTransform.to_transform_r` or
:meth:`~.HankelTransform.to_transform_k` as appropriate, and the call
:meth:`.HankelTransform.qdht` or :meth:`.HankelTransform.iqdht`.

The single-shot functions described below are demonstrated in
:ref:`sphx_glr_auto_examples_one_shot_example.py`.
For a single use, these functions are simpler, but there is a significant
overhead in creating the :class:`.HankelTransform` object, and so for
repeated transforms on the same grid, it is recommended to create the class
yourself and make repeated calls to :meth:`.HankelTransform.qdht` or
:meth:`~.HankelTransform.iqdht`.
See :ref:`sphx_glr_auto_examples_usage_example.py`
for an example demonstrating this and :ref:`sphx_glr_auto_examples_speed_usage_example.py`
for an example of the difference in speed.

.. automodule:: pyhank.one_shot
    :members: