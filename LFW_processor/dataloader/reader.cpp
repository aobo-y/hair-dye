// reader.cpp : This file finds all oroginal images of FLW dataset

#include "pch.h"
#include "reader.h"

using namespace std;
namespace fs = std::filesystem;


void Reader::reader(vector<string> global_in){
	cout << "------------------" << endl; 
	cout << "\n\tReader.cpp Started\n" << endl;

	string mask_path = global_in[0];
	string img_path = global_in[1];
	string val_path = global_in[2];
	string mask_out = global_in[3];
	string img_out = global_in[4];

	/* ------------------------ PART 1 START--------------------------------------
	read the files in ./mask directory and store them in a hashtable and a file */
	
	cout << "\tPart 1 started" << endl;

	unordered_set<string> mask_set;
	ofstream mask_names(mask_out);
	mask_names.clear();

	int mask_count = 0;	//count of files in ./mask directory

	// iterate through all mask files in ./mask directory
	for (const auto & entry : fs::directory_iterator(mask_path)){
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
		string ext = fpath.substr(dotidx+1);
		if (ext != "ppm") {
			cout << "file error! not mask file!" << endl;
			break;
		}

		string name = fpath.substr(slashidx+1, dotidx - slashidx-1);
			//cout << "\t"<<name <<endl;
		
		mask_set.insert(name);		//write the name to the hashset
		mask_names << name << endl;	//write the name to the output file

		mask_count++;
	}

	mask_names.close();
	
	// print the statistics
	cout << "\t\ttotal masks count:\t" << mask_count << endl;
	cout << "\t\tsize of mask_set: " << mask_set.size() << endl;
	cout << "\t\tfilenames are stored at:\n\t\t\t" << mask_out << endl;

	/* ------------------------ PART 1 END----------------------------------*/


	/* ------------------------ PART 2 START------------------------------
	create a new directory ./images to store the valid files in ./org */
	
	cout << "\n\tPart 2 started" << endl;

	// if ./images directory exists, erase it first
	if (fs::exists(val_path))
		fs::remove_all(val_path);

	// create ./images to store the valid images
	fs::create_directory(val_path); 
	cout << "\t\t./images directory created." << endl;

	/* ------------------------ PART 2 END----------------------------------*/


	/* ------------------------ PART 3 BEGIN----------------------------------
	find the files in ./org directory which exists in mask_set hashtable */
	
	cout << "\n\tPart 3 started" << endl;

	ofstream img_names(img_out);
	img_names.clear();

	int img_count = 0;	//count of .jpg files in ./org directory
	int match_count = 0;	//count of matched files in ./org directory

	// iterate through all files in ./org directory

	for (const auto & entry : fs::directory_iterator(img_path)) {
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
		if (ext != "jpg") {
			cout << "file error! not mask file!" << endl;
			break;
		}

		// check if current file exists in the mask_set hashtable
		string name = fpath.substr(slashidx + 1, dotidx - slashidx - 1);
			//cout << "\t"<<name <<endl;
		if (mask_set.find(name) != mask_set.end()) {
			// if current file matches, copy it to ./images directory
			fs::copy(entry.path(), val_path, fs::copy_options::overwrite_existing);

			match_count++;

			if (match_count % 200 == 0)
				cout << "\t\t..." << match_count << " files copied..." << endl;
		}

		img_count++;
	}

	img_names.close();

	// count the total number of copied files in ./images derectory
	int val_count = 0;
	for (const auto & entry : fs::directory_iterator(val_path))
		val_count++;

	// print the statistics
	cout << "\n\t\ttotal images found:\t" << img_count << endl;
	cout << "\t\ttotal images matched:\t" << match_count << endl;
	cout << "\t\ttotal images copied to ./images:\t" << val_count << endl;

	/* ------------------------ PART 3 END----------------------------------*/


	cout << "\n\tReader.cpp Finished Successfully\n" << endl;
}
