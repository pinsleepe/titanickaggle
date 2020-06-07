# Immunization Drop-outs

## Overview

This is Kedro project, which was generated using `Kedro 0.16.1` by running:

```
kedro new
```

For more info on Kedro -> [documentation](https://kedro.readthedocs.io).

## Installing dependencies

In a virtual environment with Kedro installed run:

```
kedro install
```

## Running Kedro

You can run your Kedro project with:

```
kedro run
```

## Working with Kedro from notebooks

Start a local notebook server:

```
kedro jupyter notebook
```

If you want to run an IPython session:

```
kedro ipython
```

## Building API documentation

To build API docs for your code using Sphinx, run:

```
kedro build-docs
```

See your documentation by opening `docs/build/html/index.html`.

## Building the project requirements

To generate or update the dependency requirements for your project, run:

```
kedro build-reqs
```

This will copy the contents of `src/requirements.txt` into a new file `src/requirements.in` which will be used as the source for `pip-compile`. You can see the output of the resolution by opening `src/requirements.txt`.

After this, if you'd like to update your project requirements, please update `src/requirements.in` and re-run `kedro build-reqs`.
