// dataloader.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include "pch.h"
#include "reader.h"
#include "transfer.h"
#include <iostream>
#include <vector>
#include <string>

using namespace std;

// GLOBAL definations
string mask_path = "I:/ckkr/file_reader/raw/org_masks/";
string img_path = "I:/ckkr/file_reader/raw/org_images/";
string val_path = "I:/ckkr/file_reader/raw/images/";
string mask_out = "I:/ckkr/file_reader/raw/mask_names.txt";
string img_out = "I:/ckkr/file_reader/raw/img_names.txt";
string trans_path = "I:/ckkr/file_reader/raw/p4masks/";


int main(void) {
	cout << "Program Started\n" << endl;

	// save the GLOBAL definations of path and file to vector
	vector<string> reader_in;
	reader_in.push_back(mask_path);
	reader_in.push_back(img_path);
	reader_in.push_back(val_path);
	reader_in.push_back(mask_out);
	reader_in.push_back(img_out);

	/* ------------ PART 1 START -------------
	/ call reader.cpp to prepare the images */
	Reader r;
	r.reader(reader_in);
	/* ------------ PART 1 END -------------*/


	/* ------------ PART 2 START ---------------
	/ call transfer.cpp to prepare the images */
	
	vector<string> trans_in;
	trans_in.push_back(mask_path);
	trans_in.push_back(trans_path);
	Transfer t;
	t.transPPM(trans_in);
	
	/* ------------ PART 2 END -------------*/

	cout << "\nProgram Finished Successfully\n" << endl;

	return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
