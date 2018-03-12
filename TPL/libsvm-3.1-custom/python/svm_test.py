from svmutil import *
import unittest
import filecmp


class TestSVM(unittest.TestCase):

    model_names = [
        "../data/svmguide1.model",
        "../data/svmguide1.scale.model",
    ]

    def save_model_and_check(self, model_true, model_false):
        model = svm_load_model(model_true)
        bytes_list = svm_conv_model_to_bytes(model)
        tmp_file_name = "/tmp/tmp.model"
        svm_save_model(tmp_file_name, svm_load_model_from_bytes(bytes_list))
        assert(filecmp.cmp(tmp_file_name, model_true) is True)
        assert(filecmp.cmp(tmp_file_name, model_false) is False)

    def test_save_model(self):
        self.save_model_and_check(self.model_names[0], self.model_names[1])
        self.save_model_and_check(self.model_names[1], self.model_names[0])


if __name__ == "__main__":
    unittest.main()
