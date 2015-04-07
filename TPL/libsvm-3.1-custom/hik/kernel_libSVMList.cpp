#include <string.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <list>
#include "libSVMListData.h"
#include "kernelizerList.h"

using namespace std;

bool ReadConfigFile(string &configFName, string &trainFName, string &testFName,
					string &kernelType, float &alpha, float &beta, float &rho)
{
	FILE* fconfig;	
	fconfig = fopen(configFName.c_str(), "rt");
	if (fconfig == NULL) return false;
	
	string item, value;
	char str1[255], str2[255];
	while (fscanf(fconfig, "%s%s", str1, str2)!=EOF)
	{
		item.assign(str1);
		transform(item.begin(), item.end(), item.begin(), ::tolower);
		value.assign(str2);
		if (item.compare("data_train")==0) trainFName = value;
		else if (item.compare("data_test")==0) testFName = value;
		else if (item.compare("kerneltype")==0) kernelType = value;
		else if (item.compare("alpha")==0) alpha = atof(value.c_str());
		else if (item.compare("beta")==0) beta = atof(value.c_str());
		else if (item.compare("rho")==0) rho = atof(value.c_str());
	}
	if (trainFName.empty() || testFName.empty() || kernelType.empty())
	{
		fclose(fconfig);
		return false;
	}
	
	fclose(fconfig);
	return true;
}

int main (int argc, char* argv[])
{
	if (argc!=4)
	{
		cout << "usage: kernel_libSVM <config> <outTrain> <outTest>" << endl;
		exit(-1);
	}
	string configFName = argv[1];
	string trainOutFName = argv[2];
	string testOutFName = argv[3];
	
	string trainFName, testFName;
	string kernelType;
	float alpha, beta, rho;
	ReadConfigFile(configFName, trainFName, testFName, kernelType, alpha, beta, rho);

	libSVMListData trainData(trainFName);
	cout << "Train data loaded..." << endl;
	libSVMListData testData(testFName);
	cout << "Test data loaded..." << endl;
	
	kernelizerList myKernelizer(trainData, testData);
	myKernelizer.setParam(alpha, beta, rho);
	myKernelizer.computeKernelMatrix(kernelType);
	cout << "Kernel matrix computed..." << endl;
	myKernelizer.scale();
	cout << "Kernel matrix scaled..." << endl;
	myKernelizer.saveKernelMatrix_libSVMFmt(trainOutFName, testOutFName);
	cout << "Kernel matrix saved..." << endl;
	
	return 0;
}
