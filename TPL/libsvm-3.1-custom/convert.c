
/* convert cvs data to libsvm/svm-light format */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char buf[10000000];
float feature[100000];

int main(int argc, char **argv)
{
	FILE *fp;
	
	if(argc!=2) { fprintf(stderr,"Usage %s filename\n",argv[0]); }
	if((fp=fopen(argv[1],"r"))==NULL)
	{
		fprintf(stderr,"Can't open input file %s\n",argv[1]);
	}
	
	while(fscanf(fp,"%[^\n]\n",buf)==1)
	{
		int i=0,j;
		char *p=strtok(buf,",");
		
		feature[i++]=atof(p);

		while((p=strtok(NULL,",")))
			feature[i++]=atof(p);

		//		--i;
		/*
		if ((int) feature[i]==1)
			printf("-1 ");
		else
			printf("+1 ");
		*/
		//		printf("%f ", feature[1]);
		printf("%d ", (int) feature[0]);
		for(j=1;j<i;j++)
			printf(" %d:%f",j,feature[j]);


		printf("\n");
	}
	return 0;
}
