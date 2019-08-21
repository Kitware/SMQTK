Creating a External Plugin
--------------------------
In this quick-start tutorial, we will show how to create a new interface
implementation within an external python package and expose it to the SMQTK
plugin framework via entry-points in the package's :file:`setup.py` file.

Lets assume that we are adding an implementation of the
:class:`~smqtk.algorithms.classifier.Classifier` interface to some
package we will call ``MyPackage``, wrapping the use a scikit-learn classifier
in a simple way.

Implementing the interface
^^^^^^^^^^^^^^^^^^^^^^^^^^
In :mod:`MyPackage`, lets imagine we start a new file,
:file:`new_classifier.py` such that the module is importable via the module
path :mod:`MyPackage.plugins.new_classifier`.
In the following code blocks we will incrementally build up a functional
implementation.

To start, we need to import the base interface and create a new class
inheriting from this interface:

.. code-block:: python
    :linenos:

    from sklearn.linear_model import LogisticRegression
    from smqtk.algorithms import Classifier


    class SklearnLogisticRegressionClassifier (LogisticRegression, Classifier):
        """
        A new, simple implementation of SMQTK's Classifier interface wrapping
        Scikit-Learn's LogisticRegression classifier.
        """

        @classmethod
        def is_usable(cls):
            # Required by the ``smqtk.utils.plugin.Pluggable`` parent
            return True

        def get_config(self):
            # Required by the ``smqtk.utils.configuration.Configurable`` parent.
            return {
                'C': self.C,
                'class_weight': self.class_weight,
                'dual': self.dual,
                'fit_intercept': self.fit_intercept,
                'intercept_scaling': self.intercept_scaling,
                'max_iter': self.max_iter,
                'multi_class': self.multi_class,
                'n_jobs': self.n_jobs,
                'penalty': self.penalty,
                'random_state': self.random_state,
                'solver': self.solver,
                'tol': self.tol,
                'verbose': self.verbose,
                'warm_start': self.warm_start,
            }

        def get_labels(self):
            # Required by the ``smqtk.algorithms.Classifier`` parent
            try:
                return self.classes_.tolist()
            except AttributeError:
                raise RuntimeError("No model yet fit.")

        def _classify(self, d):
            # Required by the ``smqtk.algorithms.Classifier`` parent
            proba = self.predict_proba([d.vector()])
            return zip(self.classes_, proba)

Since our source material happens to be a class itself, our implementation can
inherit from the Scikit-learn base classifier as well as from the SMQTK
interface.
In other cases, encapsulation may be a better approach.

The methods defined in our implementation are overrides of abstract methods
declared in our parent, and higher, SMQTK interfaces.
Documentation of abstract methods can usually be found in the interface sources
as doc-strings and often include what is expected to be the input and output
data-types as well as any exception conditions that are expected.
For example, the :class:`~smqtk.algorithms.classifier.Classifier` interface
documents ``get_labels`` as raising a ``RuntimeError`` specifically if no model
is loaded to access class labels.
Alternatively, :class:`~smqtk.algorithms.classifier.Classifier` documents for
the ``_classify`` method that the input parameter ``d`` will be an instance
of the :class:`~smqtk.representation.DescriptorElement` class and should return
a specifically formatted dictionary.

This implementation happens to be compliant with the defaults of the
:class:`~smqtk.utils.configuration.Configurable` interface because all of its
constructor parameters are already JSON compliant (with the occasional
exception of the "random_state" parameter when a ``RandomState`` instance is
used, but we will ignore that here for simplicity).
Thus, ``get_default_config`` will return a JSON-compliant dictionary of the
default parameters as defined in Scikit-learn's implementation, as well as
``from_config`` will appropriately return a new instance based on the given
JSON-compliant dictionary.

.. code-block:: python

    >>> dflt_config = SklearnLogisticRegressionClassifier.get_default_config()
    >>> dflt_config
    {'C': 1.0,
     'class_weight': None,
     'dual': False,
     'fit_intercept': True,
     'intercept_scaling': 1,
     'max_iter': 100,
     'multi_class': 'warn',
     'n_jobs': None,
     'penalty': 'l2',
     'random_state': None,
     'solver': 'warn',
     'tol': 0.0001,
     'verbose': 0,
     'warm_start': False}
    >>> new_dflt_inst = SklearnLogisticRegressionClassifier.from_config(dflt_config)
    >>> new_dflt_inst.get_config() == dflt_config
    True


Exposing via entry-points
^^^^^^^^^^^^^^^^^^^^^^^^^
In order to allow the SMQTK plugin framework to become aware of our new
implementation we will need to update ``MyPackage``'s :file:`setup.py` file
to add an entry-point.
Since we assumed above that we created our implementation in the module
:mod:`MyPackage.plugins.new_classifier`, the following should be added:

.. code-block:: python

    setup(
        ...
        entry_points={
            ...
            'smqtk_plugins': [
                "MyPackage_plugins = MyPackage.plugins.new_classifier",
            ]
        }
    )

Notes on adding entry-points:
  - The value to the left of the ='s sign must be unique across installed
    module providing extensions for the entry-point.
    A safe method
  - Multiple extensions may be specified.
    This may be useful if your implementations naturally belong in different
    locations within your package.
  - Currently SMQTK only supports providing modules in its extensions.
    Otherwise a warning will be emitted and that extension will be ignored.

Now, after re-installing :mod:`MyPackage`, SMQTK's plugin framework should be
able to discover this new implementation:

.. code-block:: python

    >>> from smqtk.algorithms import Classifier
    >>> classifier.get_impls()
    {..., MyPackage.plugins.new_classifier.SklearnLogisticRegressionClassifier,
     ...}
