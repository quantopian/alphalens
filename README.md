![pyfolio](https://media.quantopian.com/logos/open_source/pyfolio-logo-03.png "pyfolio")

# Qfactor
Qfactor is a Python Library for performance analysis of predictive (alpha) stock factors. Qfactor works great with the [Zipline](http://zipline.io/)
open source backtesting library, and [pyfolio](https://github.com/quantopian/pyfolio) which provides
performance and risk analysis of financial portfolios.

The main function of Qfactor is to surface the most relevant statistics and plots about an alpha factor, including:

- Information Coefficient Analysis
- Returns Analysis
- Turnover Analysis
- Sector Analysis

##Getting started
With a signal and pricing data creating a factor "tear sheet" is just:
```python
import qfactor

qfactor.tears.create_factor_tear_sheet(my_factor, pricing)
```
![](https://c1.staticflickr.com/3/2389/2073509907_345ad52bc1.jpg)

##Learn more
Check out the [example notebooks]() for more on how to read and use the factor tear sheet.

##Installation
```
pip install qfactor
```
Qfactor depends on:

- [matplotlib](https://github.com/matplotlib/matplotlib)
- [numpy](https://github.com/numpy/numpy)
- [pandas](https://github.com/pydata/pandas)
- [scipy](https://github.com/scipy/scipy)
- [seaborn](https://github.com/mwaskom/seaborn)
- [statsmodels](https://github.com/statsmodels/statsmodels)

## Usage

A good way to get started is to run the examples in a [Jupyter notebook](http://jupyter.org/).

To get set up with an example, you can:

Run a Jupyter notebook server via:

```bash
jupyter notebook
```

From the notebook list page(usually found at `http://localhost:8888/`), navigate over to the examples directory, and open any file with a .ipynb extension.

Execute the code in a notebook cell by clicking on it and hitting Shift+Enter.

## Questions?

If you find a bug, feel free to open an issue on our [github tracker](https://github.com/quantopian/qfactor/issues).

## Contribute

If you want to contribute, a great place to start would be the [help-wanted issues](https://github.com/quantopian/qfactor/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).

## Credits

* [Andrew Campbell](https://github.com/a-campbell)
* [James Christopher](https://github.com/jameschristopher)
* [Thomas Wiecki](https://github.com/twiecki)
* [Jonathan Larkin](https://github.com/marketneutral)
* Jessica Stauth (jstauth@quantopian.com)
