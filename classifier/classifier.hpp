#ifndef AIFIL_CLASSIFIER_H
#define AIFIL_CLASSIFIER_H

#include <core/raw-structures.hpp>

namespace anfisa {

struct ClassifyWindow
{
	// object size (in pixels)
	int obj_w;
	int obj_h;

	// window size
	int tile_w;
	int tile_h;

	// margins
	int margin_top;
	int margin_right;
	int margin_bottom;
	int margin_left;
};

//TODO: template class instead of typedef
typedef int classifier_input_t;
typedef void (*classifier_cb_t)(ClassifierResult *output,
	classifier_input_t *image_ptr, int rs, float sens);

struct Classifier
{
	ClassifyWindow win;

	bool valid;

	int icon_th;
	int icon_win_w;
	int icon_win_h;
};

struct ClassifierCompiled : Classifier
{
	classifier_cb_t run;
};

}  // namespace anfisa

#endif  // AIFIL_CLASSIFIER_H
