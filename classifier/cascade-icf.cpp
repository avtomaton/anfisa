#include "cascade-icf.hpp"

#include <logging.hpp>

#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include <cstdio>

namespace anfisa {

bool cascade_icf_ascent(const CascadeICF &i, const CascadeICF &j)
{
	if (i.win.tile_h < j.win.tile_h)
		return true;
	else if (i.win.tile_h == j.win.tile_h && i.win.tile_w < j.win.tile_w)
		return true;
	else
		return false;
}

CascadeICF::CascadeICF()
{
	resizable = true;
	soft_cascade = true;
	sensitivity = 0;
}

void CascadeICF::load(const std::string &fname)
{
	FILE *file = fopen(fname.c_str(), "rb");
	if (!file)
		return;

	int count = 0;
	int res = fscanf(file, "%d %d %d %d", &count, &win.tile_w, &win.tile_h, &channels);
	if (res != 4)
	{
		fclose(file);
		return;
	}
	res = fscanf(file, "%d %d %d %d",
		&win.margin_top, &win.margin_right, &win.margin_bottom, &win.margin_left);
	if (res != 4)
	{
		fclose(file);
		return;
	}
	win.obj_w = win.tile_w - win.margin_left - win.margin_right;
	win.obj_h = win.tile_h - win.margin_top - win.margin_bottom;
	bool have_error = false;
	for (int i = 0; i < count && !have_error; ++i)
	{
		DTreeICF wc;
		res = fscanf(file, "%u %e %e %e", &wc.pass, &wc.weight[0],
			&wc.weight[1], &wc.reject_threshold);
		if (res != 4)
		{
			have_error = true;
			break;
		}
		res = fscanf(file, "%d %e", &wc.features[0].count,
			&wc.features[0].min_val);
		if (res != 2)
		{
			have_error = true;
			break;
		}
		for (int q = 0; q < wc.features[0].count; q++)
		{
			res = fscanf(file, "%d %e %d %d %d %d", &wc.features[0].channel[q],
				&wc.features[0].alpha[q],
				&wc.features[0].points[q * 2].x,
				&wc.features[0].points[q * 2].y,
				&wc.features[0].points[q * 2 + 1].x,
				&wc.features[0].points[q * 2 + 1].y);
			if (res != 6)
			{
				have_error = true;
				break;
			}
		}
		if (!have_error && (wc.pass & 0x2))
		{
			res = fscanf(file, "%d %e", &wc.features[1].count,
				&wc.features[1].min_val);
			if (res != 2)
				have_error = true;
			for (int q = 0; q < wc.features[1].count; q++)
			{
				res = fscanf(file, "%d %e %d %d %d %d", &wc.features[1].channel[q],
					&wc.features[1].alpha[q],
					&wc.features[1].points[q * 2].x,
					&wc.features[1].points[q * 2].y,
					&wc.features[1].points[q * 2 + 1].x,
					&wc.features[1].points[q * 2 + 1].y);
				if (res != 6)
				{
					have_error = true;
					break;
				}
			}
		}
		if (!have_error && (wc.pass & 0x1))
		{
			res = fscanf(file, "%d %e", &wc.features[2].count,
				&wc.features[2].min_val);
			if (res != 2)
				have_error = true;
			for (int q = 0; q < wc.features[2].count; q++)
			{
				res = fscanf(file, "%d %e %d %d %d %d", &wc.features[2].channel[q],
					&wc.features[2].alpha[q],
					&wc.features[2].points[q * 2].x,
					&wc.features[2].points[q * 2].y,
					&wc.features[2].points[q * 2 + 1].x,
					&wc.features[2].points[q * 2 + 1].y);
				if (res != 6)
				{
					have_error = true;
					break;
				}
			}
		}
		weak_classifiers.push_back(wc);
	}
	fclose(file);

	if (have_error)
	{
		aifil::log_warning("Wrong cascade file!");
		weak_classifiers.clear();
	}

	valid = !!weak_classifiers.size();

	//TODO: test coefficients for orientation bins!
	if (resize_coeffs.create(channels))
		resizable = true;
	else
		resizable = false;
}

ClassifierResult CascadeICF::run(const cv::Mat &mat, int x, int y) const
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

void CascadeICF::create_scaled(float scale)
{
	if (!valid || !resizable)
	{
		aifil::log_warning("cannot create scaled ICF cascade");
		return;
	}

	win.tile_w = int(win.tile_w * scale);
	win.tile_h = int(win.tile_h * scale);
	win.obj_w = int(win.obj_w * scale);
	win.obj_w = int(win.obj_w * scale);
	win.margin_top = int(win.margin_top * scale);
	win.margin_right = int(win.margin_right * scale);
	win.margin_bottom = int(win.margin_bottom * scale);
	win.margin_left = int(win.margin_left * scale);

	for (int q = 0; q < (int)weak_classifiers.size(); ++q)
	{
		if (scale > 1.0)
			weak_classifiers[q].create_scaled(scale, &resize_coeffs.upscale_factors[0]);
		else
			weak_classifiers[q].create_scaled(scale, &resize_coeffs.downscale_factors[0]);
	}
}

void MultiscaleCascadeICF::load(const std::string &folder, const std::string &family_name)
{
	boost::filesystem::path my_dir(folder + "/");
	if (!boost::filesystem::exists(my_dir))
	{
		aifil::log_warning("classifier directory is not found");
		return;
	}
	if (!boost::filesystem::is_directory(my_dir))
	{
		aifil::log_warning("classifier directory is a regular file");
		return;
	}

	min_w = INT_MAX;
	min_h = INT_MAX;
	max_w = 0;
	max_h = 0;

	for (boost::filesystem::directory_iterator it(my_dir);
		it != boost::filesystem::directory_iterator(); ++it)
	{
		boost::filesystem::path p = it->path();
		if (!boost::filesystem::is_regular_file(p))
			continue;
		if (!boost::algorithm::contains(p.filename().string(), family_name))
			continue;
		if (p.extension().string() != ".icf")
			continue;

		std::string fname = p.generic_string();
		workers.push_back(CascadeICF());
		CascadeICF &my_model = workers.back();
		my_model.load(fname);
		if (my_model.valid)
		{
			if (my_model.win.tile_w < min_w)
				min_w = my_model.win.tile_w;
			if (my_model.win.tile_h < min_h)
				min_h = my_model.win.tile_h;
			if (my_model.win.tile_w > max_w)
				max_w = my_model.win.tile_w;
			if (my_model.win.tile_h > max_h)
				max_h = my_model.win.tile_h;
		}
		else
			workers.pop_back();
	}

	std::sort(workers.begin(), workers.end(), cascade_icf_ascent);
	valid = !workers.empty();
}

int MultiscaleCascadeICF::get_worker_index(int obj_w, int obj_h) const
{
	int res = 255;
	for (int i = 0; i < (int)workers.size(); ++i)
	{
		const CascadeICF &w = workers[i];
		if (w.win.obj_h >= obj_h && w.win.obj_w >= obj_w)
			return i;
	}
	return res;
}

} //namespace anfisa
