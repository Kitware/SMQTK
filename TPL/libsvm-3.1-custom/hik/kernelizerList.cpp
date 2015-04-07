#include <iostream>
#include <fstream>
#include <string.h>
#include <algorithm>
#include <cmath>
#include "kernelizerList.h"

using namespace std;

kernelizerList::kernelizerList(libSVMListData &trainData, libSVMListData &testData)
{
	mTrainNum = trainData.mNumSample;
	mMaxFtrDim = trainData.mMaxFtrDim;
	mlTrainData = trainData.getDataPointer();
	mTrainLabel = trainData.getLabelPointer();
	
	mTestNum = testData.mNumSample;
	if (mMaxFtrDim < testData.mMaxFtrDim) mMaxFtrDim = testData.mMaxFtrDim;
	mlTestData = testData.getDataPointer();
	mTestLabel = testData.getLabelPointer();
	
	mTrainKernel = NULL;
	mTestKernel = NULL;
	mKernelDim = 0;
	mAlpha = 0;
	mBeta = 0;
	mRho = 0;
}

kernelizerList::~kernelizerList()
{
}

void kernelizerList::setParam(float alpha, float beta, float rho)
{
	mAlpha = alpha;
	mBeta = beta;
	mRho = rho;
}

void kernelizerList::saveKernelMatrix_libSVMFmt(string trainFName, string testFName)
{
	ofstream fsTrainOut, fsTestOut;
	fsTrainOut.open(trainFName.c_str());
	if (fsTrainOut.is_open())
	{
		for (int i=0; i<mTrainNum; i++)
		{
			fsTrainOut << mTrainLabel[i] << " 0:" << i+1;
			for (int j=0; j<mKernelDim; j++)
				fsTrainOut << " " << j+1 << ":" << mTrainKernel[i][j];
			fsTrainOut << endl;
		}
		fsTrainOut.close();
	}
	
	fsTestOut.open(testFName.c_str());
	if (fsTestOut.is_open())
	{
		for (int i=0; i<mTestNum; i++)
		{
			fsTestOut << mTestLabel[i] << " 0:0";	// for a test set, we don't need to specify 0:
			for (int j=0; j<mKernelDim; j++)
				fsTestOut << " " << j+1 << ":" << mTestKernel[i][j];
			fsTestOut << endl;
		}
		fsTestOut.close();
	}
	
	return;
}

void kernelizerList::computeKernelMatrix(string kernelType)
{
	transform(kernelType.begin(), kernelType.end(), kernelType.begin(), ::tolower);
	
	if (kernelType=="hist_intsc")
		computeHistIntsc();
	else 
	{
		cout << "cannot recognize the kernel type..." << endl;
		exit(-1);
	}
	
	return;
}

void kernelizerList::memsetKernelMatrix()
{
	mTrainKernel = new float* [mTrainNum];
	mTrainKernel[0] = new float [mTrainNum*mKernelDim];
	for (int i=1; i<mTrainNum; i++)
		mTrainKernel[i] = mTrainKernel[i-1] + mKernelDim;
	memset(mTrainKernel[0], 0, sizeof(float)*mTrainNum*mKernelDim);

	mTestKernel = new float* [mTestNum];
	mTestKernel[0] = new float [mTestNum*mKernelDim];
	for (int i=1; i<mTestNum; i++)
		mTestKernel[i] = mTestKernel[i-1] + mKernelDim;
	memset(mTestKernel[0], 0, sizeof(float)*mTestNum*mKernelDim);

	return;
}

void kernelizerList::scale()
{
	float mean = 0;
	float std = 0;
	float max = -1e7;	
	for (int i=0; i<mTrainNum; i++)
	{
		for (int j=0; j<mKernelDim; j++)
		{
			mean += mTrainKernel[i][j];
			std += mTrainKernel[i][j]*mTrainKernel[i][j];
			if (max < mTrainKernel[i][j]) max = mTrainKernel[i][j];
		}
	}
	mean /= (float)(mTrainNum*mKernelDim);
	std /= (float)(mTrainNum*mKernelDim);
	std -= (mean*mean);
	std = sqrt(std);
	
	
	for (int i=0; i<mTrainNum; i++)
		for (int j=0; j<mKernelDim; j++)
	//		mTrainKernel[i][j] = (mTrainKernel[i][j] - mean)/std;
//				mTrainKernel[i][j] = mTrainKernel[i][j]/std;
			mTrainKernel[i][j] = mTrainKernel[i][j]/max;
			
	for (int i=0; i<mTestNum; i++)
		for (int j=0; j<mKernelDim; j++)
//			mTestKernel[i][j] = (mTestKernel[i][j] - mean)/std;
//			mTestKernel[i][j] = mTestKernel[i][j]/std;
			mTestKernel[i][j] = mTestKernel[i][j]/max;
	
	return;
}

void kernelizerList::computeHistIntsc()
{
	mKernelDim = mTrainNum;
	memsetKernelMatrix();
	float tmpSum;
	float max = 0;
	
	// Training Data
	for (int i=0; i<mTrainNum; i++)
	{
		//cout << "Computing training data line " << i << endl;
		for (int j=0; j<i; j++)
			mTrainKernel[i][j] = mTrainKernel[j][i];
		for (int j=i; j<mKernelDim; j++)
		{
			tmpSum = 0;
			list<svmNode>::iterator it1=mlTrainData[i].begin();
			list<svmNode>::iterator it2=mlTrainData[j].begin();
			while(it1!=mlTrainData[i].end() && it2!=mlTrainData[j].end())
			{
				if (it1->nodeNum < it2->nodeNum) it1++;
				else if (it1->nodeNum > it2->nodeNum) it2++;
				else
				{
					if (it1->value < it2->value) tmpSum += it1->value;
					else tmpSum += it2->value;
					it1++;
					it2++;
				}
			}
			mTrainKernel[i][j] = tmpSum;
		}
	}
	
	// Test Data
	for (int i=0; i<mTestNum; i++)
	{
		//cout << "Computing test data line " << i << endl;
		for (int j=0; j<mKernelDim; j++)
		{
			tmpSum = 0;
			list<svmNode>::iterator it1=mlTestData[i].begin();
			list<svmNode>::iterator it2=mlTrainData[j].begin();
			while(it1!=mlTestData[i].end() && it2!=mlTrainData[j].end())
			{
				if (it1->nodeNum < it2->nodeNum) it1++;
				else if (it1->nodeNum > it2->nodeNum) it2++;
		 		else
				{
					if (it1->value < it2->value) tmpSum += it1->value;
					else tmpSum += it2->value;
					it1++;
					it2++;
				}
			}
			mTestKernel[i][j] = tmpSum;
		}
	}
	
	return;
}

void kernelizerList::computeLinear()
{
	return;
}
