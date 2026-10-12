#include "frame_capture.h"
#include "shm_sync.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

namespace godot {

FrameCapture::FrameCapture() {}
FrameCapture::~FrameCapture() {}

void FrameCapture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("image_to_shm", "sync", "image"),
		&FrameCapture::image_to_shm);
}

bool FrameCapture::image_to_shm(ShmSync *sync, const Ref<Image> &image) {
	if (!sync || !sync->is_open()) {
		UtilityFunctions::push_warning("[FrameCapture] ShmSync 未打开");
		return false;
	}
	if (image.is_null() || image->is_empty()) {
		UtilityFunctions::push_warning("[FrameCapture] Image 无效");
		return false;
	}

	// 确保 RGBA8 格式
	Ref<Image> img = image;
	if (img->get_format() != Image::FORMAT_RGBA8) {
		img = img->duplicate();
		img->convert(Image::FORMAT_RGBA8);
	}

	PackedByteArray data = img->get_data();
	if (data.size() == 0) {
		UtilityFunctions::push_warning("[FrameCapture] Image 数据为空");
		return false;
	}

	// 直写 mmap（PackedByteArray → mmap，跳过文件 I/O）
	sync->write_raw(data.ptr(), data.size());
	return true;
}

} // namespace godot