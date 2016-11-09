#ifndef ANFISA_CASCADE_ICF_H
#define ANFISA_CASCADE_ICF_H

#include "decision-tree.hpp"
#include "classifier.hpp"

#include "core/raw-structures.hpp"
#include "feature/icf.hpp"

#include <opencv2/core/core.hpp>

#include <stdint.h>
#include <string>
#include <vector>

namespace anfisa {

struct CascadeICF : Classifier
{
	CascadeICF();
	void load(const std::string &name);

	ClassifierResult run(const cv::Mat &mat, int x, int y) const;
	void run(ClassifierResult *output, classifier_input_t *image_ptr, int rs, float sens) const;

	//input data channels num
	int channels;
	bool soft_cascade;
	float sensitivity;
	std::vector<DTreeICF> weak_classifiers;

	void create_scaled(float scale);
	//can approximate feature responses
	bool resizable;
	ResizeCoeffsICF resize_coeffs;
};

struct MultiscaleCascadeICF
{
	bool valid;
	void load(const std::string &folder, const std::string &family_name);
	ClassifierResult process(const cv::Mat &mat, int x, int y, int win_w, int win_h) const {}

	//find minimal classifier fully contains object
	int get_worker_index(int obj_w, int obj_h) const;

	//classifiers (sorted by size)
	std::vector<CascadeICF> workers;
	int min_w;
	int min_h;
	int max_w;
	int max_h;
	ClassifierResult test() {}
};

} //namespace anfisa

#endif // ANFISA_CASCADE_ICF_H
