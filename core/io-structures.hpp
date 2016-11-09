#ifndef ANFISA_IO_STRUCTURES_H
#define ANFISA_IO_STRUCTURES_H

#include <opencv2/core/core.hpp>

#include <stdint.h>
#include <list>
#include <vector>
#include <string>
#include <set>
#include <map>

namespace anfisa {

enum OBJECT_CLASS
{
	OBJECT_CLASS_UNKNOWN = 0,
	OBJECT_CLASS_HUMAN_FULLBODY = 0x1,
	OBJECT_CLASS_HUMAN_UPPERBODY = 0x2,
	OBJECT_CLASS_HUMAN_HEAD = 0x4,
	OBJECT_CLASS_HUMAN_UPVIEW = 0x8,
	OBJECT_CLASS_HUMAN_FACE = 0x10
};

struct ResultDetection
{
	ResultDetection();
	cv::Rect rect(int frame_w = 100, int frame_h = 100, int margin_w = 0, int margin_h = 0) const;
	void set_rect(const cv::Rect &rect, int frame_w, int frame_h);

	float center_x;
	float center_y;
	float width;
	float height;
	float confidence;

	int id;
	int type;
	std::string label;
	uint64_t ts;
	std::vector<uint64_t> fingerprint;
};

struct ResultTrack
{
	ResultTrack() : path_len(0) {}

	struct Point
	{
		float x;
		float y;
		uint64_t ts;
		bool processed;

		Point(float x_, float y_, uint64_t ts_)
			: x(x_), y(y_), ts(ts_), processed(false)
		{ }
	};

	typedef std::list<Point> points_t;
	points_t points;
	float path_len;
};

struct ResultTarget : public ResultDetection
{
	ResultTarget();

	float speed_x;
	float speed_y;

	bool ready;

	std::vector<uint8_t> icon;
	int icon_w;
	int icon_h;
	std::vector<cv::Point2f> shape;
	ResultTrack track;

	std::string render_figures_text(const std::string &color_rect = "",
		const std::string &color_track = "",
		const std::string &color_shape = "",
		const std::string &color_velocity = "") const;
};

struct DetectorZoneGrid
{
	DetectorZoneGrid()
		: is_valid(false), grid_w(0), grid_h(0), image_w(0), image_h(0) {}
	void render();

	bool is_valid;
	int sensitivity;
	int grid_w;
	int grid_h;
	int image_w;
	int image_h;
	std::vector<uint8_t> mask_string;
	std::vector<uint8_t> mask;
};

struct DetectorZoneParams
{
	DetectorZoneParams();

	bool is_inside(float obj_x, float obj_y) const;

	//return 0 if not crossing, direction otherwise
	int check_borders(float obj_x0, float obj_y0, float obj_x1, float obj_y1) const;
	std::string render_figures_text(const std::string &color) const;

	static const int MAX_POINTS;

	bool exists;
	std::string type; //lookup, ignore, border, border_swapped
	std::vector<double> points;

	std::string guid;
	std::string name;
	std::string folder;
	bool generate_events;
	bool t1_object;

	bool size_alarm_enabled;
	bool speed_alarm_enabled;
	bool track_alarm_enabled;

	double max_obj_size;
	double max_obj_speed;
	double max_obj_track_len;
};

struct DetectorZoneState
{
	DetectorZoneState() { reset(); }
	void update(std::vector<ResultTarget> &objects, const DetectorZoneParams &zone_params);

	//pairs of (object_id, is_new)
	typedef std::map<int, bool> obj_t;
	std::map<int, bool> cross_AB;
	std::map<int, bool> cross_BA;
	std::map<int, bool> enters;
	std::map<int, bool> leavings;
	std::map<int, bool> objects_in_zone;

	double size_w;
	double size_h;
	double size_max;
	double speed_x;
	double speed_y;
	double speed_max;
	double track_len;

	std::string color;

	void reset();
	void limits_update(const ResultTarget &obj);
	void container_sanitize(obj_t &container, const std::set<int> existing_ids);
};

}  // namespace anfisa

#endif  // ANFISA_IO_STRUCTURES_H
