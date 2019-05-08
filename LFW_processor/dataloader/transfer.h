#ifndef TRANSFER_H
#define TRANSFER_H

// transfer.h : This this the header file for transfer.cpp

#include "pch.h"
#include "ppm.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

using namespace std;

class Transfer {
public:
	void transPPM(vector<string> global_in);
};

#endif //TRANSFER_H
