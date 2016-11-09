#include "decision-tree.hpp"

#include <opencv2/imgproc/imgproc.hpp>

#include <cstdio>

namespace anfisa {

DTreeICF::DTreeICF()
	: pass(0), reject_threshold(-3.0f), approve_threshold(1000.0f)
{
}

void DTreeICF::create_scaled(float scale, const ResizeParamsICF *coeffs)
{
	features[0].create_scaled(scale, coeffs);
	if (pass & 0x1)
		features[2].create_scaled(scale, coeffs);
	if (pass & 0x2)
		features[1].create_scaled(scale, coeffs);
}

float DTreeICF::run(const cv::Mat &img, int x, int y) const
{
	float c = features[0].run(img, x, y);
	if (c > 0)
	{
		if (!(pass & 0x1))
			return weight[1];
		return weight[features[2].run(img, x, y) > 0];
	}
	else
	{
		if (!(pass & 0x2))
			return weight[0];
		return weight[features[1].run(img, x, y) > 0];
	}
}

float DTreeICF::run(const integr_img_val_t *dataxy, int rs, int ch) const
{
	float c = features[0].run(dataxy, rs, ch);
	if (c > 0)
	{
		if (!(pass & 0x1))
			return weight[1];
		return weight[features[2].run(dataxy, rs, ch) > 0];
	}
	else
	{
		if (!(pass & 0x2))
			return weight[0];
		return weight[features[1].run(dataxy, rs, ch) > 0];
	}
}

const LeafNode* CRTree::regression(uint8_t** ptFCh, int stepImg) const
{
	//pointer to current node
	const int* pnode = &treetable[0];
	int node = 0;

	//Go through tree until one arrives at a leaf, i.e. pnode[0]>=0)
	while (pnode[0] == -1)
	{
		//binary test 0 - left, 1 - right
		//Note that x, y are changed since the patches are given as matrix and not as image
		//p1 - p2 < t -> left is equal to (p1 - p2 >= t) == false

		//pointer to channel
		uint8_t* ptC = ptFCh[pnode[5]];
		// get pixel values
		int p1 = *(ptC + pnode[1] + pnode[2] * stepImg);
		int p2 = *(ptC + pnode[3] + pnode[4] * stepImg);
		// test
		bool test = (p1 - p2) >= pnode[6];

		//next node: 2 * node_id + 1 + test
		//increment node/pointer by node_id + 1 + test
		int incr = node + 1 + test;
		node += incr;
		pnode += incr * 7;
	}

	//return leaf
	return &leaf[pnode[0]];
}

bool CRTree::load(const std::string &fname)
{
	FILE *file = fopen(fname.c_str(), "rb");
	if (!file)
		return false;

	// allocate memory for tree table
	int res = fscanf(file, "%d %d %d", &max_depth, &num_leaf, &num_cp);
	if (res != 3)
	{
		fclose(file);
		return false;
	}

	bool have_error = false;
	num_nodes = (int)pow(2.0, int(max_depth + 1)) - 1;
	treetable.resize(num_nodes * 7); //num_nodes x 7 matrix as vector
	leaf.resize(num_leaf);

	//read tree nodes
	for (int n = 0; n < num_nodes && !have_error; ++n)
	{
		int dummy0 = 0;
		int dummy1 = 0;
		res = fscanf(file, "%d %d", &dummy0, &dummy1);
		if (res != 2)
		{
			have_error = true;
			break;
		}
		for (int i = 0; i < 7; ++i)
		{
			res = fscanf(file, "%d", &(treetable[7 * n + i]));
			if (res != 1)
			{
				have_error = true;
				break;
			}
		}
	}

	//read tree leafs
	for (int l = 0; l < num_leaf && !have_error; ++l)
	{
		LeafNode* ptLN = &leaf[l];
		int dummy = 0;
		res = fscanf(file, "%d %e", &dummy, &(ptLN->pfg));
		if (res != 2)
		{
			have_error = true;
			break;
		}

		//number of positive patches
		res = fscanf(file, "%d", &dummy);
		if (res != 1)
		{
			have_error = true;
			break;
		}

		ptLN->vCenter.resize(dummy);
		for (int i = 0; i < dummy && !have_error; ++i)
		{
			ptLN->vCenter[i].resize(num_cp);
			for (int k = 0; k < num_cp; ++k)
			{
				res = fscanf(file, "%d %d", &(ptLN->vCenter[i][k].x), &(ptLN->vCenter[i][k].y));
				if (res != 1)
				{
					have_error = true;
					break;
				}
			}
		}
	} //for each leaf
	fclose(file);

	valid = !have_error;
	return valid;
}

} //namespace anfisa
