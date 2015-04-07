#ifndef _EVAL_H
#define _EVAL_H

#include <stdio.h>
#include "svm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* cross validation function */
double binary_class_cross_validation(const struct svm_problem *prob, const struct svm_parameter *param, int nr_fold);

/* predict function */
void binary_class_predict(FILE *input, FILE *output); 

extern struct svm_model* model;
void exit_input_error(int line_num);

#ifdef __cplusplus
}
#endif


#endif
