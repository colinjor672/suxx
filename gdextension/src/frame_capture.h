#ifndef FRAME_CAPTURE_H
#define FRAME_CAPTURE_H

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/image.hpp>

namespace godot {

class ShmSync;

/**
 * GPU 帧捕获 — 从 Image 直写 ShmSync mmap
 *
 * 替换 Godot 的 get_image() → get_data() → store_buffer() 流程：
 *   1. GDScript 调用 get_image() 获取 GPU 渲染结果
 *   2. image_to_shm() 直接 memcpy 到 ShmSync 的 mmap 缓冲区
 *   3. 消除文件 I/O 开销（不再 open/write/close /dev/shm 文件）
 *
 * 用法（GDScript）:
 *   var capture = FrameCapture.new()
 *   var img = subviewport.get_texture().get_image()
 *   capture.image_to_shm(shm_sync, img)
 */
class FrameCapture : public Node {
	GDCLASS(FrameCapture, Node)

public:
	FrameCapture();
	~FrameCapture();

	/**
	 * 从 Image 写入 ShmSync（mmap 直写，零文件 I/O）
	 * @param sync   ShmSync 实例（Writer 模式，已 open_shm）
	 * @param image  Godot Image 对象（get_image() 结果）
	 * @return true=成功
	 */
	bool image_to_shm(ShmSync *sync, const Ref<Image> &image);

protected:
	static void _bind_methods();
};

} // namespace godot

#endif // FRAME_CAPTURE_H