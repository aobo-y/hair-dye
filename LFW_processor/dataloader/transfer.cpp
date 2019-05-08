// transfer.cpp : This file transfer the masks from *.PPM to *.PBM

#include "pch.h"
#include "ppm.h"
#include "transfer.h"

using namespace std;
namespace fs = std::filesystem;

void Transfer::transPPM(vector<string> global_in) {
	cout << "--------------------" << endl; 
	cout << "\n\ttransfer.cpp started\n" << endl;
	cout << "\t\tConvert from PNM P6 PPM to P1 PBM\n" << endl;

	string mask_path = global_in[0];
	string trans_path = global_in[1];

	/* ------------------------ PART 1 START------------------------------
	create a new directory ./masks to store the transfered files in ./org */

	cout << "\n\t\tPart 1 started" << endl;

	// if ./images directory exists, erase it first
	if (fs::exists(trans_path))
		fs::remove_all(trans_path);

	// create ./images to store the valid images
	fs::create_directory(trans_path);
	cout << "\t\t./masks directory created." << endl;

	/* ------------------------ PART 1 END----------------------------------*/


	/* ------------------------ PART 2 BEGIN----------------------------------
	transfer the *.ppm files in ./org directory */

	cout << "\n\t\tPart 2 started" << endl;

	int mask_count = 0;	//count of .ppm files in ./org directory

	// iterate through all files in ./org directory
	for (const auto & entry : fs::directory_iterator(mask_path)) {
		//cout << entry.path() << endl;

		// get the filename
		string fpath = entry.path().string();
		auto dotidx = fpath.rfind('.');
		auto slashidx = fpath.rfind('/');

		//error parsing the file name
		if (dotidx == string::npos || slashidx == string::npos) {
			cout << "error parsing the file name!" << endl;
			break;
		}
		string ext = fpath.substr(dotidx + 1);
		if (ext != "ppm") {
			cout << "file error! not mask file!" << endl;
			break;
		}

		string name = fpath.substr(slashidx + 1, dotidx - slashidx - 1);
		//cout << "\t"<<name <<endl;

		// transfer the image from *.PPM to *.PBM P1
		string fout = trans_path + name + ".pbm";

		// transfer the image from *.PPM to *.PPM P6
		//string fout = trans_path + name + ".ppm";

		PPM img;	//Initialised PPM Object
		img.open(fpath, fout); //Open PPM file
		
		mask_count++; 

		if (mask_count % 100 == 0)
			cout << "\t\t..." << mask_count << " files converted..." << endl;
	}

	// count the total number of copied files in ./images derectory
	int val_count = 0;
	for (const auto & entry : fs::directory_iterator(trans_path))
		val_count++;

	// print the statistics
	cout << "\n\t\ttotal masks found:\t" << mask_count << endl;
	cout << "\t\tnew masks added to ./masks:\t" << val_count << endl;

	/* ------------------------ PART 2 END----------------------------------*/


	cout << "\n\ttransfer.cpp Finished Successfully\n" << endl;

}