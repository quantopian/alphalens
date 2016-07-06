.. figure:: https://media.quantopian.com/logos/open_source/pyfolio-logo-03.png
   :alt: pyfolio

   pyfolio

Qfactor
=======

Qfactor is a Python Library for performance analysis of predictive
(alpha) stock factors. Qfactor works great with the
`Zipline <http://zipline.io/>`__ open source backtesting library, and
`pyfolio <https://github.com/quantopian/pyfolio>`__ which provides
performance and risk analysis of financial portfolios.

The main function of Qfactor is to surface the most relevant statistics
and plots about an alpha factor, including:

-  Information Coefficient Analysis
-  Returns Analysis
-  Turnover Analysis
-  Sector Analysis

Getting started
---------------

With a signal and pricing data creating a factor "tear sheet" is just:

.. code:: python

    import qfactor

    qfactor.tears.create_factor_tear_sheet(my_factor, pricing)

.. figure:: https://c1.staticflickr.com/3/2389/2073509907_345ad52bc1.jpg
   :alt:

Learn more
----------

Check out the `example notebooks <>`__ for more on how to read and use
the factor tear sheet.

Installation
------------

::

    pip install qfactor

Qfactor depends on:

-  `matplotlib <https://github.com/matplotlib/matplotlib>`__
-  `numpy <https://github.com/numpy/numpy>`__
-  `pandas <https://github.com/pydata/pandas>`__
-  `scipy <https://github.com/scipy/scipy>`__
-  `seaborn <https://github.com/mwaskom/seaborn>`__
-  `statsmodels <https://github.com/statsmodels/statsmodels>`__

Usage
-----

A good way to get started is to run the examples in a `Jupyter
notebook <http://jupyter.org/>`__.

To get set up with an example, you can:

Run a Jupyter notebook server via:

.. code:: bash

    jupyter notebook

From the notebook list page(usually found at
``http://localhost:8888/``), navigate over to the examples directory,
and open any file with a .ipynb extension.

Execute the code in a notebook cell by clicking on it and hitting
Shift+Enter.

Questions?
----------

If you find a bug, feel free to open an issue on our `github
tracker <https://github.com/quantopian/qfactor/issues>`__.

Contribute
----------

If you want to contribute, a great place to start would be the
`help-wanted
issues <https://github.com/quantopian/qfactor/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22>`__.

Credits
-------

-  `Andrew Campbell <https://github.com/a-campbell>`__
-  `James Christopher <https://github.com/jameschristopher>`__
-  `Thomas Wiecki <https://github.com/twiecki>`__
-  `Jonathan Larkin <https://github.com/marketneutral>`__
-  Jessica Stauth (jstauth@quantopian.com)