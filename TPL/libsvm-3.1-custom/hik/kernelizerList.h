#ifndef _KERNELIZERLIST_H
#define _KERNELIZERLIST_H

#include "libSVMListData.h"
#include <string>
#include <list>

using namespace std;

class kernelizerList
{
public:
	kernelizerList(libSVMListData &trainData, libSVMListData &testData);
	~kernelizerList();
	int mTrainNum;
	int mTestNum;
	int mMaxFtrDim;
	int mKernelDim;
	void setParam(float alpha, float beta, float rho);
	void saveKernelMatrix_libSVMFmt(string trainFName, string testFName);
	void computeKernelMatrix(string kernelType);
	void scale();
private:
	list<svmNode> *mlTrainData;
	list<svmNode> *mlTestData;
	float **mTrainKernel;
	float **mTestKernel;
	int *mTrainLabel;
	int *mTestLabel;
	float mAlpha, mBeta, mRho;
	void memsetKernelMatrix();
	void computeHistIntsc();
	void computeLinear();
};

#endif
