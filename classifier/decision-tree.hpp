#ifndef ANFISA_DECISION_TREE_H
#define ANFISA_DECISION_TREE_H

#include "feature/icf.h"

#include <opencv2/core/core.hpp>

#include <stdint.h>

namespace anfisa {

// depth-2 decision tree
struct DTreeICF
{
	uint32_t pass; //0x2 - nave left node (features[1]), 0x1 - have right node (features[2])
	FeatureVectorICF features[3];
	float weight[2];
	float reject_threshold;
	float approve_threshold;

	//scaled response: assuming run() ONLY for scaled image (approx response)
	void create_scaled(float scale, const ResizeParamsICF *coeffs);

	DTreeICF();
	float run(const cv::Mat &img, int x, int y) const;
	float run(const integr_img_val_t *dataxy, int rs, int ch) const;
};

struct IntIndex
{
	int val;
	unsigned int index;
	bool operator<(const IntIndex& a) const { return val<a.val; }
};

struct LeafNode
{
	LeafNode() {}

	float pfg; //Probability of foreground
	// Vectors from object center to training patches
	std::vector<std::vector<cv::Point> > vCenter;
};

class CRTree
{
public:
	CRTree() : valid(false) {}

	bool load(const std::string &name);

	//Set/Get functions
	int GetDepth() const { return max_depth; }
	int GetNumCenter() const { return num_cp; }

	//Regression
	const LeafNode* regression(uint8_t** ptFCh, int stepImg) const;

	bool valid;

private:

	//tree table
	//2 ^ (max_depth + 1) - 1 x 7 matrix as vector
	//column: leafindex x1 y1 x2 y2 channel thres
	//if node is not a leaf, leaf = -1
	std::vector<int> treetable;

	int min_samples; //stop growing when number of patches is less than min_samples
	int max_depth; //depth of the tree: 0-max_depth
	int num_nodes; //number of nodes: 2^(max_depth+1)-1
	int num_leaf; //number of leafs
	int num_cp; //number of center points per patch

	std::vector<LeafNode> leaf;
};

} //namespace anfisa

#endif  // ANFISA_DECISION_TREE_H
