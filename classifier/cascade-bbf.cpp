#include "cascade-bbf.hpp"

#include <logging.hpp>

#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include <cstdio>

namespace anfisa {

CascadeBBF::CascadeBBF()
	: tsr(1.0f), tsc(1.0f), tdepth(5),
	  ntrees(0)
{
}

void CascadeBBF::load(const std::string &fname)
{
}

void CascadeBBF::save(const std::string &fname)
{
}

bool CascadeBBF::load_binary(const std::string &path)
{
	FILE* file = fopen(path.c_str(), "rb");
	if (!file)
		return false;

	fread(&tsr, sizeof(float), 1, file);
	fread(&tsc, sizeof(float), 1, file);
	fread(&tdepth, sizeof(int), 1, file);

	fread(&ntrees, sizeof(int), 1, file);

	for (int i = 0; i < ntrees; ++i)
	{
		fread(&tcodes[i][0], sizeof(int32_t), (1<<tdepth)-1, file);
		fread(&luts[i][0], sizeof(float), 1<<tdepth, file);
		fread(&thresholds[i], sizeof(float), 1, file);
	}

	fclose(file);
	return true;
}

int CascadeBBF::save_binary(const std::string &path)
{
	FILE* file = fopen(path.c_str(), "wb");
	if (!file)
		return 0;

	fwrite(&tsr, sizeof(float), 1, file);
	fwrite(&tsc, sizeof(float), 1, file);
	fwrite(&tdepth, sizeof(int), 1, file);

	fwrite(&ntrees, sizeof(int), 1, file);
	for (int i = 0; i < ntrees; ++i)
	{
		fwrite(&tcodes[i][0], sizeof(int32_t), (1<<tdepth)-1, file);
		fwrite(&luts[i][0], sizeof(float), 1<<tdepth, file);
		fwrite(&thresholds[i], sizeof(float), 1, file);
	}

	fclose(file);
	return 1;
}

/*
ClassifierResult CascadeBBF::run(const cv::Mat &mat, int x, int y) const
{
	ClassifierResult res;
	run(&res, (integr_img_val_t*)mat.data, mat.step1(), sensitivity);
	return res;
}

void CascadeICF::run(ClassifierResult *res, classifier_input_t *image_ptr, int rs, float sens) const
{
	int q = 0;
	int cnt = (int)weak_classifiers.size();
	for ( ; q < cnt; ++q)
	{
		res->score += weak_classifiers[q].run(image_ptr, rs, channels);
		float rej = weak_classifiers[q].reject_threshold - sens * q / cnt;
		if (soft_cascade && res->score < rej)
		{
			res->fail = true;
			break;
		}
		if (soft_cascade && res->score > weak_classifiers[q].approve_threshold)
		{
			res->fail = false;
			break;
		}
		if (q >= 10 && q < 74)
			res->bits_desc = (1ULL << (q - 10));
	}
	res->stop_stage = q;
}
*/

}  // namespace anfisa
