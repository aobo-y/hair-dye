// ppm.cpp : *.PPM lib

#include "pch.h"
#include "ppm.h"

using namespace std;


PPM::PPM() {
	width = 0;
	height = 0;
}

void PPM::open(const string& inpath, const string& outpath) {
	// build the bit to byte map first
	/*
	bit2byte.insert({ "0000", '\x0' });
	bit2byte.insert({ "0001", '\x1' });
	bit2byte.insert({ "0010", '\x2' });
	bit2byte.insert({ "0011", '\x3' });
	bit2byte.insert({ "0100", '\x4' });
	bit2byte.insert({ "0101", '\x5' });
	bit2byte.insert({ "0110", '\x6' });
	bit2byte.insert({ "0111", '\x7' });
	bit2byte.insert({ "1000", '\x8' });
	bit2byte.insert({ "1001", '\x9' });
	bit2byte.insert({ "1010", '\xa' });
	bit2byte.insert({ "1011", '\xb' });
	bit2byte.insert({ "1100", '\xc' });
	bit2byte.insert({ "1101", '\xd' });
	bit2byte.insert({ "1110", '\xe' });
	bit2byte.insert({ "1111", '\xf' });
	*/

	ifstream fp(inpath.c_str());
	if (fp.fail()) {
		cout << "fail to open the image file!" << endl;
		return; 
	}

	//Read the Magic Number
	string mg_num, width_str, height_str, range_str;
	fp >> mg_num;
	//cout << mg_num << endl;

	// if the file is not a P6 PPM file
	if (mg_num != "P6") {
		fp.close();
		cout << "The file is not a PPM P6 file!" << endl;
		return;
	}

	fp >> width_str >> height_str >> range_str;
	width = atoi(width_str.c_str()); //Takes the number string and converts it to an integer
	height = atoi(height_str.c_str());
	unsigned int range = atoi(range_str.c_str());
	//cout << "w,h,r: " << endl;
	//cout << width << endl;
	//cout << height << endl;
	//cout << range << endl;

	//Obliterate the vector
	p6pixels.clear();
	unsigned char padding_in;
	fp >> padding_in;

	// READ P6RGB
	//Read the values into the vector directly.
	P6RGB tmp;
	int length = width * height;

	for(int i = 0; i < length; i++) {
		unsigned char inR, inG, inB;
		fp >> inR >> inG >> inB;
		tmp.R = inR;
		tmp.G = inG;
		tmp.B = inB;

		p6pixels.push_back(tmp);
	}

	int left = 0;
	unsigned char LFT;
	while (!fp.eof()) {                      // check for EOF
		left++;
		fp >> LFT;
	}
	
	//cout << "[EoF reached]£¬ with left = "<<left<<"\n";
	fp.close();

	// WRITE TO P6 PPM with RED channel available
	/*
	int wr = 0;
	// Write the header
	ofstream fout(outpath);
	fout << "P6 " << width << " " << height << " " << range<< "\n";
	// Write out each pixel to the file by grabbing its components
	while (wr < length) {
		//fout << p6pixels[wr].R;
		fout << '\x00';
		//fout << p6pixels[wr].G;
		fout << '\x00';
		fout << p6pixels[wr].B;

		wr++;
	}
	*/

	// WRITE TO P1 PBM (using '0' and '1')
	int wr = 0;
	// Write the header
	ofstream fout(outpath);
	fout << "P1 " << width << " " << height << "\n";
	// Write out each pixel to the file by grabbing its components
	while (wr < length) {
		//fout << p6pixels[wr].R;
		//fout << '\x00';
		//fout << p6pixels[wr].G;
		//fout << '\x00';
		if(p6pixels[wr].B == '\x00')
			fout << '0';
		else
			fout << '1';

		wr++;
	}
	

	fout.close();
}

/*
void PPM::readPPM6(FILE *in, P6RGB *buffer, int *width, int *height, int *maxColor) {
	unsigned char *inBuf = malloc(sizeof(unsigned char));
	vector<P6RGB> RGBbuffer;
	int i = 0;
	int conv;
	int length = (*width) * (*height);
	// while there are bytes to be read in the file 
	while (fread(inBuf, 1, 1, in) && i < length) {
		// Get each pixel into the char buffer and add to the pixel buffer
		conv = *inBuf;
		buffer[i].R = conv;
		fread(inBuf, 1, 1, in);
		conv = *inBuf;
		buffer[i].G = conv;
		fread(inBuf, 1, 1, in);
		conv = *inBuf;
		buffer[i].B = conv;
		i++;
	}
}

void writePPM6(FILE *out, pixel *buffer, int *width, int *height, int *maxColor, int *ver) {
	int i = 0;
	int length = (*width) * (*height);
	// Write the header
	fprintf(out, "P%d \n#Converted by Charles Duso\n %d %d %d\n", *ver, *width, *height, *maxColor);
	// Write out each pixel to the file by grabbing its components
	while (i < length) {
		fwrite(&(buffer[(i)].r), sizeof(unsigned char), 1, out);
		fwrite(&(buffer[(i)].g), sizeof(unsigned char), 1, out);
		fwrite(&(buffer[(i)].b), sizeof(unsigned char), 1, out);
		i++;
	}
}
*/

P6RGB& PPM::get(unsigned int a, unsigned int b) {
	//return pixels[(b * width) + a];
	return p6pixels[(b * width) + a];
}

unsigned int PPM::getWidth() {
	return width;
}

unsigned int PPM::getHeight() {
	return height;
}