// ppm.cpp : *.PPM lib

#include "pch.h"
#include "ppm.h"
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace std;


PPM::PPM() {
	width = 0;
	height = 0;
}

void PPM::readPPM6(FILE *in, pixel *buffer, int *width, int *height, int *maxColor) {
	unsigned char *inBuf = malloc(sizeof(unsigned char));
	vector<p6pixels> RGBbuffer;
	int i = 0;
	int conv;
	int length = (*width) * (*height);
	/* while there are bytes to be read in the file */
	while (fread(inBuf, 1, 1, in) && i < length) {
		/* Get each pixel into the char buffer and add to the pixel buffer */
		conv = *inBuf;
		buffer[i].r = conv;
		fread(inBuf, 1, 1, in);
		conv = *inBuf;
		buffer[i].g = conv;
		fread(inBuf, 1, 1, in);
		conv = *inBuf;
		buffer[i].b = conv;
		i++;
	}
}

void PPM::open(const string& path) {
	ifstream fp(path.c_str());
	if (fp.fail()) {
		cout << "fail to open the image file!" << endl;
		return; 
	}

	//Read the Magic Number
	string mg_num, width_str, height_str, range_str;
	fp >> mg_num;
	cout << mg_num << endl;

	// if the file is not a ASCII PPM file
	if (mg_num != "P6") {
		fp.close();
		cout << "The file is not a PPM P6 file!" << endl;
		return;
	}

	fp >> width_str >> height_str >> range_str;
	width = atoi(width_str.c_str()); //Takes the number string and converts it to an integer
	height = atoi(height_str.c_str());
	unsigned int range = atoi(range_str.c_str());
	cout << "w,h,r: " << endl;
	cout << width << endl;
	cout << height << endl;
	cout << range << endl;

	//Obliterate the vector
	pixels.clear();

	//Read the values into the vector directly.
	RGB tmp;
	string _R, _G, _B;
	for (unsigned int i = 0; i < width * height; i++) {
		fp >> _R >> _G >> _B;
		tmp.R = atoi(_R.c_str());
		tmp.G = atoi(_G.c_str());
		tmp.B = atoi(_B.c_str());

		pixels.push_back(tmp);
		
		int red, green, blue;
		unsigned int pixel;
		/* packing */
		pixel = 0xff << 24 | blue << 16 | green << 8 | red;
		/* unpacking */
		alpha = (pixel >> 24);
		blue = (pixel >> 16) & 0xff;
		green = (pixel >> 8) & 0xff;
		red = pixel & 0xff;
	}

	fp.close();
}

RGB& PPM::get(unsigned int a, unsigned int b) {
	return pixels[(b * width) + a];
}

unsigned int PPM::getWidth() {
	return width;
}

unsigned int PPM::getHeight() {
	return height;
}