#ifndef SHM_SYNC_H
#define SHM_SYNC_H

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>

namespace godot {

/**
 * POSIX 共享内存 + 信号量同步
 *
 * 替换 /dev/shm 文件 I/O，实现零拷贝共享内存通信：
 *   - mmap: 直接映射共享内存，避免 open/read/write 系统调用
 *   - semaphore: 信号量替代轮询，CPU 零开销等待新帧
 *
 * 双缓冲协议：
 *   - Writer 写入当前槽 → post semaphore
 *   - Reader wait semaphore → 读取当前槽
 *   每帧数据格式: [4字节 frame_id:u32 LE][像素数据]
 */
class ShmSync : public Node {
	GDCLASS(ShmSync, Node)

public:
	ShmSync();
	~ShmSync();

	// 打开/创建共享内存
	// @param name    共享内存名称（如 "shm_input"）
	// @param size    数据区大小（字节）
	// @param is_writer  true=Writer(创建+写入), false=Reader(只读)
	bool open_shm(const String &name, int64_t size, bool is_writer);

	// 关闭共享内存和信号量
	void close_shm();

	// 读取共享内存数据（Reader 调用）
	// 返回 mmap 的零拷贝视图（PackedByteArray 引用 mmap 内存）
	PackedByteArray get_data() const;

	// 写入共享内存数据（Writer 调用）
	void store_data(const PackedByteArray &data);

	// 直接写入原始内存（C++ 内部调用，避免 PackedByteArray 中间拷贝）
	// FrameCapture 使用此方法将 GPU 像素数据直写 mmap
	void write_raw(const uint8_t *src, int64_t size);

	// 获取数据区写指针（C++ 内部使用，glReadPixels 直接写入）
	uint8_t *get_write_ptr() { return _data_ptr(); }

	// 读取帧 ID（前 4 字节）
	uint32_t get_frame_id() const;

	// 写入帧 ID（前 4 字节）
	void set_frame_id(uint32_t id);

	// 等待新帧（Reader 调用，阻塞直到 Writer post 或超时）
	// @param timeout_ms  超时毫秒，0=无限等待，-1=非阻塞
	// @return true=有新帧, false=超时
	bool wait_for_new_frame(int timeout_ms = 0);

	// 通知新帧就绪（Writer 调用）
	void signal_new_frame();

	// 是否已打开
	bool is_open() const { return _mmap_ptr != nullptr; }

	// 获取数据区大小
	int64_t get_size() const { return _data_size; }

protected:
	static void _bind_methods();

private:
	int _shm_fd = -1;
	void *_mmap_ptr = nullptr;
	int64_t _total_size = 0;   // 总大小 = 4字节头 + 数据区
	int64_t _data_size = 0;    // 数据区大小
	String _shm_name;
	bool _is_writer = false;

	// POSIX 信号量
	void *_sem = nullptr;       // sem_t*
	String _sem_name;

	// 清理信号量
	void _cleanup_sem();

	// 获取数据区指针（跳过 4 字节帧 ID 头）
	uint8_t *_data_ptr() const {
		return static_cast<uint8_t *>(_mmap_ptr) + 4;
	}

	// 获取帧 ID 指针
	uint32_t *_frame_id_ptr() const {
		return static_cast<uint32_t *>(_mmap_ptr);
	}
};

} // namespace godot

#endif // SHM_SYNC_H