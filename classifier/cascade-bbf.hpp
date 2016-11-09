#ifndef ANFISA_CASCADE_BBF_H
#define ANFISA_CASCADE_BBF_H

#include "decision-tree.hpp"
#include "classifier.hpp"

#include "core/raw-structures.hpp"
#include "feature/icf.hpp"

#include <opencv2/core/core.hpp>

#include <stdint.h>
#include <string>
#include <vector>

namespace anfisa {

struct CascadeBBF : Classifier
{
	CascadeBBF();
	void load(const std::string &name);
	void save(const std::string &name);
	void load_binary(const std::string &name);
	void save_binary(const std::string &name);

	/*
	ClassifierResult run(const cv::Mat &mat, int x, int y) const;
	void run(ClassifierResult *output, classifier_input_t *image_ptr, int rs, float sens) const;
	*/

	float tsr;
	float tsc;
	int tdepth;
	int ntrees;

	int32_t tcodes[4096][1024];
	float luts[4096][1024];
	float thresholds[4096];
};

} //namespace anfisa

#endif  // ANFISA_CASCADE_BBF_H
