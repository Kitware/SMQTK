#include "libSVMListData.h"
#include <fstream>
#include <iostream>
#include <algorithm>

libSVMListData::libSVMListData(string dataFName)
{
	mNumSample = countSample(dataFName);
	mDataList = new list<svmNode> [mNumSample];
	mLabel = new int [mNumSample];
	mMaxFtrDim = 0;
	loadData(dataFName);
}

libSVMListData::~libSVMListData()
{
	if (mDataList) delete [] mDataList;
	if (mLabel) delete [] mLabel;
}

int libSVMListData::countSample(string dataFName)
{
	cout << "count sample in " << endl;
	int numSample = 0;

	ifstream fsData;
	fsData.open(dataFName.c_str());
	if (!fsData.is_open())
	{
		cout << "Cannot open " << dataFName << endl;
		exit(-1);
	}
	string line_buf;
	while(!fsData.eof())
	{
		getline(fsData, line_buf);
		if (line_buf.empty() || line_buf[0]=='\n') break;
		numSample++;
	}
	fsData.close();	

	cout << "count sample out " << endl;
	return numSample;
}

void libSVMListData::loadData(string dataFName)
{
	cout << "load data in" << endl;
	ifstream fsData;
	fsData.open(dataFName.c_str());
	if (!fsData.is_open())
	{
		cout << "Cannot open " << dataFName << endl;
		exit(-1);
	}
	string line_buf, dummy_str;
	int strIndex1, strIndex2;
	svmNode tmpNode;
	for (int i=0; i<mNumSample; i++)
	{
		getline(fsData, line_buf);
		strIndex2 = line_buf.find(" ");
		dummy_str = line_buf.substr(0, strIndex2);
		mLabel[i] = atoi(dummy_str.c_str());

		strIndex1 = strIndex2 + 1;
		while(strIndex1 < line_buf.length())
		{
			strIndex2 = line_buf.find(":", strIndex1);
			if (strIndex2 < 0) break;

			dummy_str = line_buf.substr(strIndex1, strIndex2-strIndex1);
			tmpNode.nodeNum = atoi(dummy_str.c_str());
			strIndex1 = line_buf.find(" ", strIndex2+1);
			dummy_str = line_buf.substr(strIndex2+1, strIndex1-(strIndex2+1));
			tmpNode.value = atof(dummy_str.c_str());
			if (tmpNode.nodeNum > 0)
				mDataList[i].push_back(tmpNode);

			if (strIndex1 < 0) break;
			strIndex1 = strIndex1 + 1;
			if (mMaxFtrDim < tmpNode.nodeNum) mMaxFtrDim = tmpNode.nodeNum;
		}
	}
	list<svmNode>::iterator it = mDataList[0].end();
	it--;
	cout << it->value << endl;
	
	cout << "load data out" << endl;
	return;
}

void libSVMListData::saveData(string dataFName)
{
	ofstream fsData;
	fsData.open(dataFName.c_str());
	if (!fsData.is_open()) 
	{
		cout << "Cannot open " << dataFName << endl;
		exit(-1);
	}
	for (int i=0; i<mNumSample; i++)
	{
		fsData << mLabel[i];
		for (list<svmNode>::iterator it=mDataList[i].begin(); it!=mDataList[i].end(); it++)
		{
			fsData << " " << it->nodeNum << ":" << it->value;
		}
		fsData << endl;
	}
	fsData.close();
	
	return;
}
