"""
LICENCE
-------
Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

"""

class ECDStoreElement (object):
    """
    Encapsulation of an element stored in the ECD Store. Consists of the results
    of an ECD process and the metadata to uniquely identify the result.
    """

    def __init__(self, model_id, clip_id, probability):
        """
        Construct a ECDStoreElement.

        :param model_id: The id of the model used to generate the result.
        :type model_id: str
        :param clip_id: The clip the result is the measurement of.
        :type clip_id: int
        :param probability: The probability result of the model to the clip.
        :type probability: float

        """
        self.model_id = str(model_id)
        self.clip_id = int(clip_id)
        self.probability = float(probability)

    def __repr__(self):
        return "ECDStoreElement{model_id: '%s', clip_id: %d, probability: %f}" \
            % (self.model_id, self.clip_id, self.probability)
