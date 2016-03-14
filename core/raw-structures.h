#ifndef ANFISA_RAW_STRUCTURES_H
#define ANFISA_RAW_STRUCTURES_H

#include <stdint.h>

//please no external lib dependencies here!

namespace anfisa {

struct Classifier;

struct ClassifierResult
{
	float score;
	bool fail;
	int stop_stage;
	uint64_t bits_desc;

	ClassifierResult() : score(0), fail(false), stop_stage(0), bits_desc(0) {}
};

struct DetectionRaw
{
	int id;
	int x;
	int y;
	int width;
	int height;
	float confidence;
	int neighbours;
	int scale_n;
	uint64_t fingerprint;

	DetectionRaw()
		: id(0), confidence(0), neighbours(0) {}
};

struct MinMaxSize
{
	int min_w;
	int min_h;
	int max_w;
	int max_h;
};

struct CUDAClassifier
{
	int *data_ptr;
	int data_size;
	bool data_loaded;
	ClassifierResult *res_ptr;
	int res_size;

	CUDAClassifier()
		: data_ptr(0), data_size(0), data_loaded(false), res_ptr(0), res_size(0) {}
	//TODO: ~CUDAClassifier() { cudaFree(data_ptr); cudaFree(res_ptr); }
};

}  // namespace anfisa

#endif  // ANFISA_RAW_STRUCTURES_H
