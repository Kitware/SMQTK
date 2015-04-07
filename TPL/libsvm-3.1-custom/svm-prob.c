#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

void exit_with_help()
{
  printf(
  "Usage: svm-prob data_filename hist_filename\n"
  );
  exit(1);
}

char *line = NULL;
int max_line_len = 10240;
double lower=-1.0,upper=1.0,y_lower,y_upper;
int y_scaling = 0;
double *feature_max;
double *feature_min;
double y_max = -DBL_MAX;
double y_min = DBL_MAX;
int max_index;
long int num_nonzeros = 0;
long int new_num_nonzeros = 0;

#define max(x,y) g(((x)>(y))?(x):(y))
#define min(x,y) (((x)<(y))?(x):(y))

void output_target(double value);
void output(int index, double value);
char* readline(FILE *input);

int main(int argc,char **argv)
{
  int i,index;
  FILE *fp, *fp_hist = NULL;

  if( argc < 3 )
  {
    fprintf(stderr, "usage: %s data_file hist_file\n", argv[0]);
    exit(-1);
  }
  
  fp=fopen(argv[1],"r");
  if(fp==NULL)
  {
    fprintf(stderr,"can't open data file %s\n", argv[1]);
    exit(1);
  }
  
  fp_hist=fopen(argv[2],"w");
  if(fp_hist==NULL)
  {
    fprintf(stderr,"can't open histogram file %s\n", argv[2]);
    exit(1);
  }
  fp_hist=fopen(argv[2],"w");
  
  line = (char *) malloc(max_line_len*sizeof(char));

#define SKIP_TARGET\
  while(isspace(*p)) ++p;\
  while(!isspace(*p)) ++p;

#define SKIP_ELEMENT\
  while(*p!=':') ++p;\
  ++p;\
  while(isspace(*p)) ++p;\
  while(*p && !isspace(*p)) ++p;
  
  max_index = 0;

  /* pass 3: scale */
  while(readline(fp)!=NULL)
  {
    char *p=line;
    int next_index=1;
    double target;
    double value;
    
    sscanf(p,"%lf",&target);
/*    output_target(target);*/
    fprintf(fp_hist,"%06d ", (int)(target+0.5));
    
    SKIP_TARGET

    float sum=0.0;
    while(sscanf(p,"%d:%lf",&index,&value)==2)
    {
      sum += value;
      SKIP_ELEMENT
      next_index=index+1;
    }

    p=line;
    sscanf(p,"%lf",&target);
    SKIP_TARGET
    
    while(sscanf(p,"%d:%lf",&index,&value)==2)
    {
      /*
      for(i=next_index;i<index;i++)
        output(i,0);
      */
/*      output(index,value/sum);*/
      fprintf(fp_hist, "%d:%g ",index, value/sum);
      SKIP_ELEMENT
      next_index=index+1;
    }   

    /*
    for(i=next_index;i<=max_index;i++)
      output(i,0);
    */
    fprintf(fp_hist, "\n");
  }

  free(line);
  fclose(fp);
  fclose(fp_hist);
  return 0;
}

char* readline(FILE *input)
{
  int len;
  
  if(fgets(line,max_line_len,input) == NULL)
    return NULL;

  while(strrchr(line,'\n') == NULL)
  {
    max_line_len *= 2;
    line = (char *) realloc(line, max_line_len);
    len = (int) strlen(line);
    if(fgets(line+len,max_line_len-len,input) == NULL)
      break;
  }
  return line;
}

void output_target(double value)
{
  printf("%g ",value);
}

void output(int index, double value)
{
  /* skip single-valued attribute */
  if(feature_max[index] == feature_min[index])
    return;

  if(value == feature_min[index])
    value = lower;
  else if(value == feature_max[index])
    value = upper;
  else
    value = lower + (upper-lower) * 
      (value-feature_min[index])/
      (feature_max[index]-feature_min[index]);

  if(value != 0)
  {
    printf("%d:%g ",index, value);
    new_num_nonzeros++;
  }
}
