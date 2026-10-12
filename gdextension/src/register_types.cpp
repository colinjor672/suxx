#include "register_types.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>

#include "shm_sync.h"
#include "frame_capture.h"

using namespace godot;

void initialize_shm_video_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	ClassDB::register_class<ShmSync>();       // mmap + semaphore 共享内存
	ClassDB::register_class<FrameCapture>();   // GPU 帧抓取直写 mmap
}

void uninitialize_shm_video_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
}

extern "C" {
	GDExtensionBool GDE_EXPORT
	shm_video_init(GDExtensionInterfaceGetProcAddress p_get_proc_address,
	               GDExtensionClassLibraryPtr p_library,
	               GDExtensionInitialization *r_initialization) {
		godot::GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);
		init_obj.register_initializer(initialize_shm_video_module);
		init_obj.register_terminator(uninitialize_shm_video_module);
		init_obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);
		return init_obj.init();
	}
}