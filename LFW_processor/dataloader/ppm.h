#ifndef PPM_H
#define PPM_H

// ppm.h : This this the header file for ppm.cpp

#include "pch.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <bitset>
#include <vector>
#include <string>
#include <map>

using namespace std;

struct P6RGB {
	//Values
	unsigned char R, G, B;
};

class PPM {
public:
	PPM();
	void open(const string& infile, const string& outfile);
	P6RGB& get(unsigned int, unsigned int);
	void binaryToText(char *binary, int binaryLength, char *text, int symbolCount);
	unsigned int getWidth();
	unsigned int getHeight();

private:
	unsigned int width, height;
	vector<P6RGB> p6pixels;
	map<string, unsigned char> bit2byte;
};

#endif //PPM_H
