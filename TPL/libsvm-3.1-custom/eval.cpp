#include <iostream>
#include <vector>
#include <algorithm>
#include <errno.h>
#include <cstring>
#include "svm.h"
#include "eval.h"
#include <stdlib.h>

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

typedef std::vector<double> dvec_t;
typedef std::vector<int>    ivec_t;

// prototypes of evaluation functions
double precision(const dvec_t& dec_values, const ivec_t& ty);
double recall(const dvec_t& dec_values, const ivec_t& ty);
double fscore(const dvec_t& dec_values, const ivec_t& ty);
double bac(const dvec_t& dec_values, const ivec_t& ty);
double auc(const dvec_t& dec_values, const ivec_t& ty);
double accuracy(const dvec_t& dec_values, const ivec_t& ty);

// evaluation function pointer
// You can assign this pointer to any above prototype
double (*validation_function)(const dvec_t&, const ivec_t&) = auc;


static char *line = NULL;
static int max_line_len;


static char* readline(FILE *input)
{
  int len;
  
  if(fgets(line,max_line_len,input) == NULL)
    return NULL;

  while(strrchr(line,'\n') == NULL)
  {
    max_line_len *= 2;
    line = (char *) realloc(line,max_line_len);
    len = (int) strlen(line);
    if(fgets(line+len,max_line_len-len,input) == NULL)
      break;
  }
  return line;
}



double precision(const dvec_t& dec_values, const ivec_t& ty){
  size_t size = dec_values.size();
  size_t i;
  int    tp, fp;
  double precision;

  tp = fp = 0;

  for(i = 0; i < size; ++i)
    if(dec_values[i] >= 0)
    {
      if(ty[i] == 1)
        ++tp;
      else
        ++fp;
    }

  if(tp + fp == 0){
    fprintf(stderr, "warning: No postive predict label.\n");
    precision = 0;
  }else
    precision = tp / (double) (tp + fp);
  printf("Precision = %g%% (%d/%d)\n", 100.0 * precision, tp, tp + fp);
  
  return precision;
}



double recall(const dvec_t& dec_values, const ivec_t& ty){
  size_t size = dec_values.size();
  size_t i;
  int    tp, fn; // true_positive and false_negative
  double recall;

  tp = fn = 0;

  for(i = 0; i < size; ++i) if(ty[i] == 1){ // true label is 1
    if(dec_values[i] >= 0) ++tp; // predict label is 1
    else                   ++fn; // predict label is -1
  }

  if(tp + fn == 0){
    fprintf(stderr, "warning: No postive true label.\n");
    recall = 0;
  }else
    recall = tp / (double) (tp + fn);
  // print result in case of invocation in prediction
  printf("Recall = %g%% (%d/%d)\n", 100.0 * recall, tp, tp + fn);
  
  return recall; // return the evaluation value
}



double fscore(const dvec_t& dec_values, const ivec_t& ty){
  size_t size = dec_values.size();
  size_t i;
  int    tp, fp, fn;
  double precision, recall;
  double fscore;

  tp = fp = fn = 0;

  for(i = 0; i < size; ++i) 
    if(dec_values[i] >= 0 && ty[i] == 1) ++tp;
    else if(dec_values[i] >= 0 && ty[i] == -1) ++fp;
    else if(dec_values[i] <  0 && ty[i] == 1) ++fn;

  if(tp + fp == 0){
    fprintf(stderr, "warning: No postive predict label.\n");
    precision = 0;
  }else
    precision = tp / (double) (tp + fp);
  if(tp + fn == 0){
    fprintf(stderr, "warning: No postive true label.\n");
    recall = 0;
  }else
    recall = tp / (double) (tp + fn);

  
  if(precision + recall == 0){
    fprintf(stderr, "warning: precision + recall = 0.\n");
    fscore = 0;
  }else
    fscore = 2 * precision * recall / (precision + recall);

  printf("F-score = %g\n", fscore);
  
  return fscore;
}



double bac(const dvec_t& dec_values, const ivec_t& ty){
  size_t size = dec_values.size();
  size_t i;
  int    tp, fp, fn, tn;
  double specificity, recall;
  double bac;

  tp = fp = fn = tn = 0;

  for(i = 0; i < size; ++i) 
    if(dec_values[i] >= 0 && ty[i] == 1) ++tp;
    else if(dec_values[i] >= 0 && ty[i] == -1) ++fp;
    else if(dec_values[i] <  0 && ty[i] == 1)  ++fn;
    else ++tn;

  if(tn + fp == 0){
    fprintf(stderr, "warning: No negative true label.\n");
    specificity = 0;
  }else
    specificity = tn / (double)(tn + fp);
  if(tp + fn == 0){
    fprintf(stderr, "warning: No positive true label.\n");
    recall = 0;
  }else
    recall = tp / (double)(tp + fn);

  bac = (specificity + recall) / 2;
  printf("BAC = %g\n", bac);
  
  return bac;
}



// only for auc
class Comp{
  const double *dec_val;
  public:
  Comp(const double *ptr): dec_val(ptr){}
  bool operator()(int i, int j) const{
    return dec_val[i] > dec_val[j];
  }
};


double auc(const dvec_t& dec_values, const ivec_t& ty){
  double roc  = 0;
  size_t size = dec_values.size();
  size_t i;
  std::vector<size_t> indices(size);

  for(i = 0; i < size; ++i) indices[i] = i;

///*/*  std::sort(indices.begin(), indices.end(), Comp(&dec_values[0]));

  int tp = 0,fp = 0;
  for(i = 0; i < size; i++) {
    if(ty[indices[i]] == 1) tp++;
    else if(ty[indices[i]] == -1) {
      roc += tp;
      fp++;
    }
  }

  if(tp == 0 || fp == 0)
  {
    fprintf(stderr, "warning: Too few postive true labels or negative true labels\n");
    roc = 0;
  }
  else
    roc = roc / tp / fp;

  printf("AUC = %g\n", roc);

  return roc;
}



double accuracy(const dvec_t& dec_values, const ivec_t& ty){
  int    correct = 0;
  int    total   = (int) ty.size();
  size_t i;

  for(i = 0; i < ty.size(); ++i)
    if(ty[i] == (dec_values[i] >= 0? 1: -1)) ++correct;

  printf("Accuracy = %g%% (%d/%d)\n",
    (double)correct/total*100,correct,total);

  return (double) correct / total;
}



double binary_class_cross_validation(const svm_problem *prob, const svm_parameter *param, int nr_fold)
{
  int i;
  int *fold_start = Malloc(int,nr_fold+1);
  int l = prob->l;
  int *perm = Malloc(int,l);
  int *labels;
  dvec_t dec_values;
  ivec_t ty;

  for(i=0;i<l;i++) perm[i]=i;
  for(i=0;i<l;i++)
  {
    int j = i+rand()%(l-i);
    std::swap(perm[i],perm[j]);
  }
  for(i=0;i<=nr_fold;i++)
    fold_start[i]=i*l/nr_fold;

  printf("%d-fold cross validation\n",nr_fold);
  char *home=getenv("HOME");
  char fname[2048];
  sprintf(fname, "%s/binary_class_cross_valdation.dat", home);
  printf("written detection to file: %s\n", fname);
  FILE *fp=fopen(fname,"w");
  
  for(i=0;i<nr_fold;i++)
  {
    int                begin   = fold_start[i];
    int                end     = fold_start[i+1];
    int                j,k;
    struct svm_problem subprob;

    subprob.l = l-(end-begin);
    subprob.x = Malloc(struct svm_node*,subprob.l);
    subprob.y = Malloc(double,subprob.l);

    k=0;
    for(j=0;j<begin;j++)
    {
      subprob.x[k] = prob->x[perm[j]];
      subprob.y[k] = prob->y[perm[j]];
      ++k;
    }
    for(j=end;j<l;j++)
    {
      subprob.x[k] = prob->x[perm[j]];
      subprob.y[k] = prob->y[perm[j]];
      ++k;
    }
    struct svm_model *submodel = svm_train(&subprob,param);
    int svm_type = svm_get_svm_type(submodel);
  
    if(svm_type == NU_SVR || svm_type == EPSILON_SVR){
      fprintf(stderr, "wrong svm type");
      exit(1);
    }

    labels = Malloc(int, svm_get_nr_class(submodel));
    svm_get_labels(submodel, labels);

    if(svm_get_nr_class(submodel) > 2) 
    {
      fprintf(stderr,"Error: the number of class is not equal to 2\n");
      exit(-1);
    }

    dec_values.resize(end);
    ty.resize(end);

    for(j=begin;j<end;j++) {
      binary_predict_values(submodel,prob->x[perm[j]], &dec_values[j]);
      fprintf(fp, "%g\n", dec_values[j]);

      svm_predict_values(submodel,prob->x[perm[j]], &dec_values[j]);
      ty[j] = (prob->y[perm[j]] > 0)? 1: -1;

      
//      int index=prob->y[perm[j]];
//      fprintf(fp, "%03d\t%g\n", index, dec_values[j]);
    }

    // output cross-validation prediction scores
//    for(j=begin;j<end;j++)
//      fprintf(fp, "%g\n", dec_values[j]);
    
/*
    if(labels[0] <= 0) {
      for(j=begin;j<end;j++)
        dec_values[j] *= -1;
    }
*/  
   
    svm_free_and_destroy_model(&submodel);
    free(subprob.x);
    free(subprob.y);
    free(labels);
  }   

  int total_correct=0;
  for(i=0;i<prob->l;i++)
    if(ty[i] == prob->y[i])
      ++total_correct;
  printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob->l);

  fclose(fp);
  free(perm);
  free(fold_start);

  return validation_function(dec_values, ty); 
}


void binary_class_predict(FILE *input, FILE *output){
  int    total = 0;
  int    *labels;
  int    max_nr_attr = 64;
  struct svm_node *x = Malloc(struct svm_node, max_nr_attr);
  dvec_t dec_values;
  ivec_t true_labels;


  int svm_type=svm_get_svm_type(model);
  
  if (svm_type==NU_SVR || svm_type==EPSILON_SVR){
    fprintf(stderr, "wrong svm type.");
    exit(1);
  }

  labels = Malloc(int, svm_get_nr_class(model));
  svm_get_labels(model, labels);
  
  max_line_len = 1024;
  line = (char *)malloc(max_line_len*sizeof(char));
  while(readline(input) != NULL)
  {
    int i = 0;
    double target_label, predict_label;
    char *idx, *val, *label, *endptr;
    int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0

    label = strtok(line," \t");
    target_label = strtod(label,&endptr);

//    printf("%g \n", target_label);
    
    if(endptr == label)
      exit_input_error(total+1);

    while(1)
    {
      if(i>=max_nr_attr - 2)  // need one more for index = -1
      {
        max_nr_attr *= 2;
        x = (struct svm_node *) realloc(x,max_nr_attr*sizeof(struct svm_node));
      }

      idx = strtok(NULL,":");
      val = strtok(NULL," \t");

      if(val == NULL)
        break;
      errno = 0;
      x[i].index = (int) strtol(idx,&endptr,10);
      if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
        exit_input_error(total+1);
      else
        inst_max_index = x[i].index;

      errno = 0;
      x[i].value = strtod(val,&endptr);
      if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
        exit_input_error(total+1);

      ++i;
    }
    x[i].index = -1;

//    predict_label = binary_predict(model,x);
//    fprintf(output,"%g\n",predict_label);
//    printf("%g  ",predict_label);


    double dec_value;
    predict_label = binary_predict_values(model, x, &dec_value);
//    printf("%g \t %g \t %g\n", predict_label, dec_value, target_label);
    fprintf(output,"%g\n",predict_label);

    true_labels.push_back((target_label > 0)? 1: -1);
//    if(labels[0] <= 0) dec_value *= -1;
    dec_values.push_back(dec_value);
  } 

//  for( unsigned int b=0; b<dec_values.size(); b++)
//    printf("%d %d %f \n", b, true_labels[b], dec_values[b]);
  
  validation_function(dec_values, true_labels);

  /*  */
  accuracy(dec_values, true_labels);
  precision(dec_values, true_labels);
  recall(dec_values, true_labels);
  /*  */

  free(labels);
  free(x);
}
