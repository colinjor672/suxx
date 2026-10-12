#include "shm_sync.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <semaphore.h>
#include <cerrno>
#include <cstring>
#include <ctime>

namespace godot {

ShmSync::ShmSync() {}

ShmSync::~ShmSync() {
	close_shm();
}

void ShmSync::_bind_methods() {
	ClassDB::bind_method(D_METHOD("open_shm", "name", "size", "is_writer"), &ShmSync::open_shm);
	ClassDB::bind_method(D_METHOD("close_shm"), &ShmSync::close_shm);
	ClassDB::bind_method(D_METHOD("get_data"), &ShmSync::get_data);
	ClassDB::bind_method(D_METHOD("store_data", "data"), &ShmSync::store_data);
	ClassDB::bind_method(D_METHOD("get_frame_id"), &ShmSync::get_frame_id);
	ClassDB::bind_method(D_METHOD("set_frame_id", "id"), &ShmSync::set_frame_id);
	ClassDB::bind_method(D_METHOD("wait_for_new_frame", "timeout_ms"), &ShmSync::wait_for_new_frame, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("signal_new_frame"), &ShmSync::signal_new_frame);
	ClassDB::bind_method(D_METHOD("is_open"), &ShmSync::is_open);
	ClassDB::bind_method(D_METHOD("get_size"), &ShmSync::get_size);
}

bool ShmSync::open_shm(const String &name, int64_t size, bool is_writer) {
	close_shm(); // 先关闭旧的

	_shm_name = name;
	_data_size = size;
	_total_size = size + 4; // 4 字节帧 ID 头
	_is_writer = is_writer;

	// 1. 打开/创建共享内存
	int oflag = O_RDWR;
	if (is_writer) {
		oflag |= O_CREAT;
	}

	_shm_fd = shm_open(name.utf8().get_data(), oflag, 0666);
	if (_shm_fd < 0) {
		UtilityFunctions::printerr("[ShmSync] shm_open(\"", name, "\") 失败: ", strerror(errno));
		_shm_fd = -1;
		return false;
	}

	// 2. Writer 设置大小
	if (is_writer) {
		if (ftruncate(_shm_fd, _total_size) < 0) {
			UtilityFunctions::printerr("[ShmSync] ftruncate 失败: ", strerror(errno));
			::close(_shm_fd);
			_shm_fd = -1;
			return false;
		}
	}

	// 3. mmap
	_mmap_ptr = mmap(nullptr, _total_size, PROT_READ | PROT_WRITE, MAP_SHARED, _shm_fd, 0);
	if (_mmap_ptr == MAP_FAILED) {
		UtilityFunctions::printerr("[ShmSync] mmap 失败: ", strerror(errno));
		_mmap_ptr = nullptr;
		::close(_shm_fd);
		_shm_fd = -1;
		return false;
	}

	// 4. 打开/创建信号量
	_sem_name = String("/") + name + "_sem";
	if (is_writer) {
		// Writer: 创建信号量，初始值为 0
		_sem = sem_open(_sem_name.utf8().get_data(), O_CREAT, 0666, 0);
	} else {
		// Reader: 打开已有信号量
		_sem = sem_open(_sem_name.utf8().get_data(), 0);
	}

	if (_sem == SEM_FAILED) {
		UtilityFunctions::printerr("[ShmSync] sem_open(\"", _sem_name, "\") 失败: ", strerror(errno));
		_sem = nullptr;
		// 不致命，可以继续使用 mmap（回退到轮询模式）
	}

	UtilityFunctions::print("[ShmSync] ", is_writer ? "Writer" : "Reader",
		" \"", name, "\" 已打开, 数据区 ", String::num_int64(size), " 字节, ",
		_sem ? "semaphore OK" : "semaphore FAIL (回退轮询)");

	return true;
}

void ShmSync::close_shm() {
	_cleanup_sem();

	if (_mmap_ptr) {
		munmap(_mmap_ptr, _total_size);
		_mmap_ptr = nullptr;
	}

	if (_shm_fd >= 0) {
		::close(_shm_fd);
		_shm_fd = -1;
	}

	// Writer 清理时删除共享内存对象
	if (_is_writer && !_shm_name.is_empty()) {
		shm_unlink(_shm_name.utf8().get_data());
	}

	_data_size = 0;
	_total_size = 0;
	_is_writer = false;
}

void ShmSync::_cleanup_sem() {
	if (_sem) {
		sem_close(static_cast<sem_t *>(_sem));
		_sem = nullptr;
	}
	// Writer 清理时删除信号量
	if (_is_writer && !_sem_name.is_empty()) {
		sem_unlink(_sem_name.utf8().get_data());
	}
}

PackedByteArray ShmSync::get_data() const {
	PackedByteArray arr;
	if (!_mmap_ptr || _data_size <= 0) {
		return arr;
	}
	arr.resize(_data_size);
	memcpy(arr.ptrw(), _data_ptr(), _data_size);
	return arr;
}

void ShmSync::store_data(const PackedByteArray &data) {
	if (!_mmap_ptr) {
		return;
	}
	int64_t copy_size = data.size();
	if (copy_size > _data_size) {
		copy_size = _data_size;
		UtilityFunctions::push_warning("[ShmSync] store_data 数据过大, 截断 ",
			String::num_int64(data.size()), " -> ", String::num_int64(_data_size));
	}
	memcpy(_data_ptr(), data.ptr(), copy_size);
}

uint32_t ShmSync::get_frame_id() const {
	if (!_mmap_ptr) return 0;
	return *_frame_id_ptr();
}

void ShmSync::write_raw(const uint8_t *src, int64_t size) {
	if (!_mmap_ptr) return;
	int64_t copy_size = size;
	if (copy_size > _data_size) {
		copy_size = _data_size;
	}
	memcpy(_data_ptr(), src, copy_size);
}

void ShmSync::set_frame_id(uint32_t id) {
	if (!_mmap_ptr) return;
	*_frame_id_ptr() = id;
}

bool ShmSync::wait_for_new_frame(int timeout_ms) {
	if (!_sem) {
		// 无信号量：非阻塞轮询模式，当前帧已就绪
		return true;
	}

	sem_t *sem = static_cast<sem_t *>(_sem);

	if (timeout_ms < 0) {
		// 非阻塞：trywait
		return sem_trywait(sem) == 0;
	}

	if (timeout_ms == 0) {
		// 无限等待
		while (sem_wait(sem) != 0) {
			if (errno != EINTR) {
				return false;
			}
		}
		return true;
	}

	// 带超时等待
	struct timespec ts;
	clock_gettime(CLOCK_REALTIME, &ts);
	ts.tv_sec  += timeout_ms / 1000;
	ts.tv_nsec += (timeout_ms % 1000) * 1000000;
	if (ts.tv_nsec >= 1000000000) {
		ts.tv_sec  += 1;
		ts.tv_nsec -= 1000000000;
	}

	while (sem_timedwait(sem, &ts) != 0) {
		if (errno == ETIMEDOUT) {
			return false;
		}
		if (errno != EINTR) {
			return false;
		}
	}
	return true;
}

void ShmSync::signal_new_frame() {
	if (!_sem) return;
	sem_post(static_cast<sem_t *>(_sem));
}

} // namespace godot