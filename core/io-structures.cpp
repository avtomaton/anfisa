#include "io-structures.h"

#include <common/errutils.h>
#include <common/stringutils.h>

#include <opencv2/imgproc/imgproc.hpp>

namespace anfisa {

const int DetectorZoneParams::MAX_POINTS = 50;

ResultDetection::ResultDetection()
	: confidence(0), type(OBJECT_CLASS_UNKNOWN), ts(0)
{
	id = 0;
	type = 0;
	center_x = 0;
	center_y = 0;
	width = 0;
	height = 0;
}

cv::Rect ResultDetection::rect(int frame_w, int frame_h, int margin_w, int margin_h) const
{
	int left = std::max(0, int(frame_w * (center_x - width / 2) / 100.0f + 0.5f) - margin_w);
	int top = std::max(0, int(frame_h * (center_y - height / 2) / 100.0f + 0.5f) - margin_h);
	int right = std::min(int(frame_w * (center_x + width / 2) / 100.0f + 0.5f) + margin_w, frame_w - 1);
	int bottom = std::min(int(frame_h * (center_y + height / 2) / 100.0f + 0.5f) + margin_h, frame_h - 1);
	return cv::Rect(left, top, right - left, bottom - top);
}

void ResultDetection::set_rect(const cv::Rect &rect, int frame_w, int frame_h)
{
	center_x = (rect.x + rect.width / 2) * 100.0f / frame_w;
	center_y = (rect.y + rect.height / 2) * 100.0f / frame_h;
	width = rect.width * 100.0f / frame_w;
	height = rect.height * 100.0f / frame_h;
}

ResultTarget::ResultTarget()
	: ready(false), icon_w(0), icon_h(0)
{
}

std::string ResultTarget::render_figures_text(const std::string &color_rect,
	const std::string &color_track,
	const std::string &color_shape,
	const std::string &color_velocity) const
{
	std::string fig;
	if (!color_rect.empty())
	{
		fig += aifil::stdprintf("rect %0.2f %0.2f %0.2f %0.2f %s",
			center_x - width / 2, center_y - height / 2,
			center_x + width / 2, center_y + height / 2,
			color_rect.c_str());
		fig += std::string("", 1);
	}
	if (!color_track.empty() && track.points.size() > 2)
	{
		std::list<ResultTrack::Point>::const_iterator tr_it = track.points.begin();
		std::list<ResultTrack::Point>::const_iterator tr_next = tr_it;
		++tr_next;
		for (; tr_next != track.points.end(); ++tr_it, ++tr_next)
		{
			const ResultTrack::Point& p0 = *tr_it;
			const ResultTrack::Point& p1 = *tr_next;
			fig += aifil::stdprintf("line %0.2lf %0.2lf %0.2lf %0.2lf %s",
				p0.x, p0.y, p1.x, p1.y, color_track.c_str());
			fig += std::string("", 1);
		}
		const ResultTrack::Point& p0 = *tr_it;
		fig += aifil::stdprintf("line %0.2lf %0.2lf %0.2lf %0.2lf %s",
			p0.x, p0.y, center_x, center_y, color_track.c_str());
		fig += std::string("", 1);
	}
	if (!color_shape.empty() && shape.size() > 2)
	{
		for (size_t j = 0; j < shape.size() - 1; ++j)
		{
			const cv::Point2f &p0 = shape[j];
			const cv::Point2f &p1 = shape[j + 1];
			fig += aifil::stdprintf("line %0.2f %0.2f %0.2f %0.2f %s",
				p0.x, p0.y, p1.x, p1.y, color_shape.c_str());
			fig += std::string("", 1);
		}
		//contour must be closed
		const cv::Point2f &p0 = shape[0];
		const cv::Point2f &p1 = shape[shape.size() - 1];
		fig += aifil::stdprintf("line %0.2lf %0.2lf %0.2lf %0.2lf %s",
			p0.x, p0.y, p1.x, p1.y, color_shape.c_str());
		fig += std::string("", 1);
	}
	if (!color_velocity.empty())
	{
		fig += aifil::stdprintf("line %0.2f %0.2f %0.2f %0.2f %s",
			center_x, center_y,
			center_x + 30 * speed_x,
			center_y + 30 * speed_y,
			color_velocity.c_str());
		fig += std::string("", 1);
	}
	return fig;
}

void DetectorZoneGrid::render()
{
	af_assert(image_w && image_h && grid_w && grid_h);
	af_assert(grid_w < image_w && grid_h < image_h);

	mask.resize(image_w * image_h, 0);

	for (int y = 0; y < image_h; ++y)
		for (int x = 0; x < image_w; ++x)
			mask[y * image_w + x] =
				mask_string[grid_w * (grid_h * y / image_h) + grid_w * x / image_w];

	is_valid = true;
}

DetectorZoneParams::DetectorZoneParams()
{
	exists = false;
	t1_object = false;
	generate_events = false;
	type = "lookup";

	size_alarm_enabled = false;
	speed_alarm_enabled = false;
	track_alarm_enabled = false;
	max_obj_size = 0;
	max_obj_speed = 0;
	max_obj_track_len = 0;
}

bool DetectorZoneParams::is_inside(float obj_x, float obj_y) const
{
	if (type != "lookup")
		return false;

	std::vector<cv::Point> pts;
	for (size_t j = 0; j < points.size() / 2; j++)
		pts.push_back(cv::Point(int(points[2 * j]), int(points[2 * j + 1])));

	double measure = cv::pointPolygonTest(pts, cv::Point2f(obj_x, obj_y), false);
	if (measure >= 0)
		return true;

	return false;
}

int DetectorZoneParams::check_borders(float obj_x0, float obj_y0, float obj_x1, float obj_y1) const
{
	if (type == "ignore")
		return 0;

	for (int point_num = 0; point_num <= int(points.size()) / 2 - 2; ++point_num)
	{
		float x1 = float(points[2 * point_num]);
		float y1 = float(points[2 * point_num + 1]);
		float x2 = float(points[2 * point_num + 2]);
		float y2 = float(points[2 * point_num + 3]);

		//d = (x2 - x1)*(y3 - y4) - (x3 - x4)*(y2 - y1);

		if ((((obj_x0 - x1) * (y2 - y1) - (obj_y0 - y1) * (x2 - x1)) *
			((obj_x1 - x1) * (y2 - y1) - (obj_y1 - y1) * (x2 - x1)) <= 0) &&
			(((x1 - obj_x0) * (obj_y1 - obj_y0) - (y1 - obj_y0) * (obj_x1 - obj_x0)) *
			((x2 - obj_x0) * (obj_y1 - obj_y0) - (y2 - obj_y0) * (obj_x1 - obj_x0)) <= 0))
		{
			if (type == "lookup")
				return 1;

			bool left2right = (obj_x0 - x1) * (y2 - y1) - (obj_y0 - y1) * (x2 - x1) < 0;
			if (type == "border" && left2right)
				return 1;
			else if (type == "border")
				return -1;
			else if (type == "border_swapped" && left2right)
				return -1;
			else if (type == "border_swapped")
				return 1;

			af_assert(!"incorrect zone type");
		}
	}

	//last point
	if (type == "lookup")
	{
		int point_num = points.size() - 2;
		float x1 = float(points[point_num]);
		float y1 = float(points[point_num + 1]);
		float x2 = float(points[0]);
		float y2 = float(points[1]);

		if ((((obj_x0 - x1) * (y2 - y1) - (obj_y0 - y1) * (x2 - x1)) *
			((obj_x1 - x1) * (y2 - y1) - (obj_y1 - y1) * (x2 - x1)) <= 0) &&
			(((x1 - obj_x0) * (obj_y1 - obj_y0) - (y1 - obj_y0) * (obj_x1 - obj_x0)) *
			((x2 - obj_x0) * (obj_y1 - obj_y0) - (y2 - obj_y0) * (obj_x1 - obj_x0)) <= 0))
		{
			return 1;
		}
	}

	return 0;
}

std::string DetectorZoneParams::render_figures_text(const std::string &color) const
{
	std::string fig;
	if (!exists)
		return fig;

	int size = points.size();
	if (size % 2)
		size--;
	if (size < 4)
		return fig;

	if (size > 2 * MAX_POINTS)
		size = 2 * MAX_POINTS;

	int index_max_len = 0;
	double max_len = 0;
	for (int j = 0; j < size / 2 - 1; ++j)
	{
		fig += aifil::stdprintf("line %0.2lf %0.2lf %0.2lf %0.2lf %s",
			points[2 * j], points[2 * j + 1], points[2 * j + 2], points[2 * j + 3],
			color.c_str());
		fig += std::string("", 1);
		double len = (points[2 * j + 2] - points[2 * j]) *
			(points[2 * j + 2] - points[2 * j]) +
			(points[2 * j + 3] - points[2 * j + 1]) *
			(points[2 * j + 3] - points[2 * j + 1]);
		if (len > max_len)
		{
			max_len = len;
			index_max_len = 2 * j;
		}
	}

	if (type != "border")
	{
		fig += aifil::stdprintf("line %0.2lf %0.2lf %0.2lf %0.2lf %s",
			points[size - 2], points[size - 1], points[0], points[1], color.c_str());
		fig += std::string("", 1);
	}
	else if (max_len > 0)
	{
		max_len = sqrt(max_len);
		double x0 = points[index_max_len];
		double y0 = points[index_max_len + 1];
		double x1 = points[index_max_len + 2];
		double y1 = points[index_max_len + 3];
		double xa = (x0 + x1) / 2 - (y0 - y1) * 4 / max_len;
		double ya = (y0 + y1) / 2 - (x1 - x0) * 5 / max_len;
		double xb = (x0 + x1) / 2 + (y0 - y1) * 4 / max_len;
		double yb = (y0 + y1) / 2 + (x1 - x0) * 5 / max_len;
		fig += aifil::stdprintf("text %0.2lf %0.2lf %0.2lf %0.2lf %s A",
			xa, ya - 3, xa + 4, ya + 3, color.c_str());
		fig += std::string("", 1);
		fig += aifil::stdprintf("text %0.2lf %0.2lf %0.2lf %0.2lf %s B",
			xb, yb - 3, xb + 4, yb + 3, color.c_str());
		fig += std::string("", 1);
	}
	return fig;
}

void DetectorZoneState::reset()
{
	objects_in_zone.clear();
	size_w = 0;
	size_h = 0;
	size_max = 0;
	speed_x = 0;
	speed_y = 0;
	speed_max = 0;
	track_len = 0;
}

void DetectorZoneState::update(
	std::vector<ResultTarget> &objects, const DetectorZoneParams &zone_params)
{
	reset();
	bool is_border = (zone_params.type == "border" || zone_params.type == "border_swapped");

	for (size_t tg_ind = 0; tg_ind < objects.size(); ++tg_ind)
	{
		ResultTarget &object = objects[tg_ind];
		bool inside = zone_params.is_inside(object.center_x, object.center_y);
		if (inside)
		{
			objects_in_zone[object.id] = true;
			limits_update(object);
		}

		if (object.track.points.size() < 2 || object.track.points.back().processed)
			continue;

		ResultTrack::points_t::reverse_iterator cur_pt = object.track.points.rbegin();
		ResultTrack::points_t::reverse_iterator prev_pt = cur_pt++;
		while (prev_pt != object.track.points.rend() && !cur_pt->processed)
		{
			int cross = zone_params.check_borders(prev_pt->x, prev_pt->y, cur_pt->x, cur_pt->y);
			if (!cross)
				continue;

			if (is_border)
			{
				if (cross > 0)
					cross_AB[object.id] = true;
				else
					cross_BA[object.id] = true;
				limits_update(object);
			}
			else
			{
				if (inside)
					enters[object.id] = true;
				else
					leavings[object.id] = true;
			}

			cur_pt->processed = true;
			++prev_pt;
			++cur_pt;
		} //for each track point
	} //for each object

	//remove lost objects
	std::set<int> existing;
	for (size_t i = 0; i < objects.size(); ++i)
		existing.insert(objects[i].id);
	container_sanitize(cross_AB, existing);
	container_sanitize(cross_BA, existing);
	container_sanitize(enters, existing);
	container_sanitize(leavings, existing);
}

void DetectorZoneState::limits_update(const ResultTarget &obj)
{
	if (obj.width > size_w)
		size_w = obj.width;
	if (obj.height > size_h)
		size_h = obj.height;
	size_max = std::max(size_w, size_h);

	double speed = sqrt(obj.speed_x * obj.speed_x + obj.speed_y * obj.speed_y);
	if (speed > speed_max)
	{
		speed_max = speed;
		speed_x = obj.speed_x;
		speed_y = obj.speed_y;
	}

	if (obj.track.path_len > track_len)
		track_len = obj.track.path_len;
}

void DetectorZoneState::container_sanitize(obj_t &container, const std::set<int> existing_ids)
{
	std::set<int> victims;
	for (obj_t::iterator it = container.begin(); it != container.end(); ++it)
	{
		if (existing_ids.find(it->first) == existing_ids.end())
			victims.insert(it->first);
	}

	for (std::set<int>::const_iterator it1 = victims.begin(); it1 != victims.end(); ++it1)
		container.erase(*it1);
}

}  // namespace anfisa
