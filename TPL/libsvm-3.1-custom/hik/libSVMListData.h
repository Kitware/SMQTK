#ifndef _LIBSVMLISTDATA_H
#define _LIBSVMLISTDATA_H

#include <list>
#include <string>
#include "svmNode.h"

using namespace std;

class libSVMListData
{
public:
	libSVMListData(string dataFName);
	~libSVMListData();
	int mNumSample;
	int mMaxFtrDim;
	void saveData(string dataFName);
	list<svmNode>* getDataPointer() {return mDataList;};
	int* getLabelPointer() {return mLabel;};
private:
	list<svmNode> *mDataList;
	int *mLabel;
	int countSample(string dataFName);
	void loadData(string dataFName);
};

#endif
