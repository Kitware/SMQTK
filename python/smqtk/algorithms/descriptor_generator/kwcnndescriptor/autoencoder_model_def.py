"""
Auto-encoder network model definition. Only defined if kwcnn is available.
"""
from .kwcnndescriptor import kwcnn


AutoEncoderModel = None
if kwcnn is not None:
    class AutoEncoderModel(kwcnn.core.KWCNN_Auto_Model):  # NOQA
        """FCNN Model."""

        def __init__(self, *args, **kwargs):
            """FCNN init."""
            super(AutoEncoderModel, self).__init__(*args, **kwargs)
            self.greyscale = kwargs.get("greyscale", False)
            self.bottleneck = kwargs.get("bottleneck", 64)
            self.trimmed = kwargs.get("trimmed", False)

        def _input_shape(self):
            return (64, 64, 1) if self.greyscale else (64, 64, 3)

        # noinspection PyProtectedMember
        def architecture(self, batch_size, in_width, in_height,
                         in_channels, out_classes):
            """FCNN architecture."""
            input_height, input_width, input_channels = self._input_shape()
            _nonlinearity = kwcnn.tpl._lasagne.nonlinearities.LeakyRectify(
                leakiness=(1. / 10.)
            )

            #: :type: list[lasagne.layers.Layer]
            layer_list = [
                kwcnn.tpl._lasagne.layers.InputLayer(
                    shape=(None, input_channels, input_height, input_width)
                )
            ]

            for index in range(2):
                layer_list.append(
                    kwcnn.tpl._lasagne.layers.batch_norm(
                        kwcnn.tpl._lasagne.Conv2DLayer(
                            layer_list[-1],
                            num_filters=16,
                            filter_size=(3, 3),
                            nonlinearity=_nonlinearity,
                            W=kwcnn.tpl._lasagne.init.Orthogonal(),
                        )
                    )
                )

            layer_list.append(
                kwcnn.tpl._lasagne.MaxPool2DLayer(
                    layer_list[-1],
                    pool_size=(2, 2),
                    stride=(2, 2),
                )
            )

            for index in range(3):
                layer_list.append(
                    kwcnn.tpl._lasagne.layers.batch_norm(
                        kwcnn.tpl._lasagne.Conv2DLayer(
                            layer_list[-1],
                            num_filters=32,
                            filter_size=(3, 3),
                            nonlinearity=_nonlinearity,
                            W=kwcnn.tpl._lasagne.init.Orthogonal(),
                        )
                    )
                )

            layer_list.append(
                kwcnn.tpl._lasagne.MaxPool2DLayer(
                    layer_list[-1],
                    pool_size=(2, 2),
                    stride=(2, 2),
                )
            )

            for index in range(2):
                layer_list.append(
                    kwcnn.tpl._lasagne.layers.batch_norm(
                        kwcnn.tpl._lasagne.Conv2DLayer(
                            layer_list[-1],
                            num_filters=32,
                            filter_size=(3, 3),
                            nonlinearity=_nonlinearity,
                            W=kwcnn.tpl._lasagne.init.Orthogonal(),
                        )
                    )
                )

            l_reshape0 = kwcnn.tpl._lasagne.layers.ReshapeLayer(
                layer_list[-1],
                shape=([0], -1),
            )

            if self.trimmed:
                return l_reshape0

            l_bottleneck = kwcnn.tpl._lasagne.layers.DenseLayer(
                l_reshape0,
                num_units=self.bottleneck,
                nonlinearity=kwcnn.tpl._lasagne.nonlinearities.tanh,
                W=kwcnn.tpl._lasagne.init.Orthogonal(),
                name="bottleneck",
            )

            return l_bottleneck
