//
// Created by chaturvedi on 2/20/18.
//

#include <gtest/gtest.h>
#include "svm.h"
#include <cmath>


class SVMTest : public ::testing::Test
{
 protected:
  svm_model *sample_model1;
  svm_model *sample_model2;

  virtual void SetUp() {
    sample_model1 = svm_load_model("./data/svmguide1.model");
    sample_model2 = svm_load_model("./data/svmguide1.scale.model");
  }
  virtual void TearDown(){
    svm_free_and_destroy_model(&sample_model1);
    svm_free_and_destroy_model(&sample_model2);
  }

 public:
  // TODO (Mmanu) : Suboptimal because we're saving a file and diff-ing it
  // But letting it be because we're just testing
  bool check_models_equal(svm_model* m, const char* orig_model_file_name){
    const char *tmp_file_name = "/tmp/tmp_svm.model";
    const char *diff_file = "/tmp/tmp_diff.op";

    svm_save_model(tmp_file_name, m);
    char buff[100];
    // Use diff to make sure that the models written are exactly the same
    sprintf(buff, "diff %s %s > %s", tmp_file_name, orig_model_file_name, diff_file);
    system(buff);

    // Check that the diff output is an empty file
    FILE *fp = fopen(diff_file, "r");
    // Got to the 0th place before end.
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    bool res = false;

    fclose(fp);
    if(size == 0) res = true;
    std::remove(diff_file);
    std::remove(tmp_file_name);
    return res;
  }

  void test_model(const char *original_model_name, const char *other_model_name){
    svm_model* original_model = svm_load_model(original_model_name);


    unsigned char* p;
    unsigned long size;
    // Load and check the model from bytes array
    convert_model_to_bytes(original_model, p, size);
    svm_model* returned_model = load_model_from_bytes(p, size);
    ASSERT_TRUE(check_models_equal(returned_model, original_model_name));
    ASSERT_FALSE(check_models_equal(returned_model, other_model_name));
    free(p);
    svm_free_and_destroy_model(&returned_model);

    // Load and check the model from bytes vector
    std::vector<unsigned char>* p_vec = convert_model_to_bytes_vector(original_model);
    svm_model* returned_model_vector = load_model_from_bytes_vector(p_vec);
    ASSERT_TRUE(check_models_equal(returned_model_vector, original_model_name));
    ASSERT_FALSE(check_models_equal(returned_model_vector, other_model_name));
    delete p_vec;
    svm_free_and_destroy_model(&returned_model_vector);
    svm_free_and_destroy_model(&original_model);
  }

};

TEST_F(SVMTest, checkFirstModel) {
  const char* testing_model_name = "./data/svmguide1.model";
  const char* other_model_name = "./data/svmguide1.scale.model";
  test_model(testing_model_name, other_model_name);
}

TEST_F(SVMTest, checkSecondModel) {
  const char* testing_model_name = "./data/svmguide1.scale.model";
  const char* other_model_name = "./data/svmguide1.model";
  test_model(testing_model_name, other_model_name);
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
