.. include:: references.txt

.. _installation:

************
Installation
************

Requirements
============

**astroplan** works on Linux, Mac OS X, and Windows.
It requires Python 3.11+ as well as numpy, astropy, and pytz.
Additional features are available when you install `Matplotlib`_
and `astroquery`_.

Installation
============

You can install the stable version of astroplan from PyPI with::

    pip install astroplan

or from anaconda::

    conda install -c conda-forge astroplan

Alternatively, you can install the latest developer version of astroplan by
cloning the git repository::

    git clone https://github.com/astropy/astroplan

...then installing the package with::

    cd astroplan
    pip install .

Testing
=======

If you want to check that all the tests are running correctly with your Python
configuration, run the following from the command line::

    tox -e test

If there are no errors, you are good to go!

.. note::
	If you want to run the tests that access the internet, you'll need to
	replace the last line above with ``tox -e test -- --remote-data`` and
	have an active connection to the internet.  Also, if you want the tests
	that check plotting to work, you need `Matplotlib`_ and `pytest-mpl`_.

More
====

astroplan follows `astropy <https://astropy.org>`__'s guidelines for affiliated packages--installation
and testing for the two are quite similar! Please see
:ref:`astropy's installation page <astropy:installing-astropy>`
for more information.
