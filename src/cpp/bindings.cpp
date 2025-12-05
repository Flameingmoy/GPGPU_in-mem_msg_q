/**
 * @file bindings.cpp
 * @brief PyBind11 bindings for GPUQueue (Track B)
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "gpuqueue/queue.hpp"
#include "gpuqueue/gpu_queue.hpp"
#include "gpuqueue/types.hpp"

namespace py = pybind11;
using namespace gpuqueue;

PYBIND11_MODULE(_core, m) {
    m.doc() = "GPUQueue: CUDA GPU-resident message queue";

    // === Legacy functions (for backward compatibility) ===
    m.def("version", []() { return gpuqueue::version(); },
          "Return version string of the gpuqueue core");

    m.def("init", &gpuqueue::init, py::arg("device") = 0,
          "Initialize the gpuqueue core (legacy, prefer GpuQueue class)");

    m.def("shutdown", &gpuqueue::shutdown,
          "Shutdown the gpuqueue core (legacy)");

    // === QueueStatus enum ===
    py::enum_<QueueStatus>(m, "QueueStatus", "Status codes for queue operations")
        .value("OK", QueueStatus::OK, "Operation succeeded")
        .value("ERR_INVALID_ARG", QueueStatus::ERR_INVALID_ARG, "Invalid argument")
        .value("ERR_FULL", QueueStatus::ERR_FULL, "Queue is full")
        .value("ERR_TIMEOUT", QueueStatus::ERR_TIMEOUT, "Operation timed out")
        .value("ERR_CUDA", QueueStatus::ERR_CUDA, "CUDA error")
        .value("ERR_SHUTDOWN", QueueStatus::ERR_SHUTDOWN, "Queue is shut down")
        .value("ERR_NOT_READY", QueueStatus::ERR_NOT_READY, "Result not ready yet")
        .value("ERR_NOT_FOUND", QueueStatus::ERR_NOT_FOUND, "Message not found")
        .value("ERR_NOT_INITIALIZED", QueueStatus::ERR_NOT_INITIALIZED, "Queue not initialized")
        .value("ERR_ALREADY_INITIALIZED", QueueStatus::ERR_ALREADY_INITIALIZED, "Queue already initialized")
        .value("ERR_PAYLOAD_TOO_LARGE", QueueStatus::ERR_PAYLOAD_TOO_LARGE, "Payload exceeds slot size")
        .export_values();

    // === QueueConfig struct ===
    py::class_<QueueConfig>(m, "QueueConfig", "Configuration for GpuQueue")
        .def(py::init<>())
        .def(py::init([](uint32_t capacity, uint32_t slot_bytes) {
            QueueConfig cfg;
            cfg.capacity = capacity;
            cfg.slot_bytes = slot_bytes;
            return cfg;
        }), py::arg("capacity") = 1024, py::arg("slot_bytes") = 512,
           "Create queue config with capacity (power of 2) and slot size")
        .def_readwrite("capacity", &QueueConfig::capacity,
                      "Number of slots (must be power of 2)")
        .def_readwrite("slot_bytes", &QueueConfig::slot_bytes,
                      "Size of each slot in bytes")
        .def_readwrite("num_blocks", &QueueConfig::num_blocks,
                      "Number of CUDA blocks for kernel")
        .def_readwrite("num_threads", &QueueConfig::num_threads,
                      "Number of threads per block")
        .def("is_valid", &QueueConfig::is_valid,
             "Check if configuration is valid")
        .def("__repr__", [](const QueueConfig& c) {
            return "QueueConfig(capacity=" + std::to_string(c.capacity) +
                   ", slot_bytes=" + std::to_string(c.slot_bytes) + ")";
        });

    // === QueueStats struct ===
    py::class_<QueueStats>(m, "QueueStats", "Statistics from GpuQueue")
        .def(py::init<>())
        .def_readonly("enqueue_count", &QueueStats::enqueue_count,
                     "Total messages enqueued")
        .def_readonly("dequeue_count", &QueueStats::dequeue_count,
                     "Total messages dequeued")
        .def_readonly("processed_count", &QueueStats::processed_count,
                     "Total messages processed by kernel")
        .def_readonly("error_count", &QueueStats::error_count,
                     "Processing errors")
        .def_readonly("dropped_full", &QueueStats::dropped_full,
                     "Messages dropped due to full queue")
        .def_readonly("queue_depth", &QueueStats::queue_depth,
                     "Current queue depth")
        .def_readonly("head", &QueueStats::head, "Current head position")
        .def_readonly("tail", &QueueStats::tail, "Current tail position")
        .def("__repr__", [](const QueueStats& s) {
            return "QueueStats(enqueued=" + std::to_string(s.enqueue_count) +
                   ", dequeued=" + std::to_string(s.dequeue_count) +
                   ", processed=" + std::to_string(s.processed_count) +
                   ", depth=" + std::to_string(s.queue_depth) + ")";
        });

    // === GpuQueue class ===
    py::class_<GpuQueue>(m, "GpuQueue", R"doc(
        GPU-resident message queue (Track B).
        
        Messages are enqueued from the host, stored in GPU VRAM, processed by
        a persistent CUDA kernel, and results returned to the host.
        
        Example:
            q = GpuQueue(QueueConfig(capacity=1024, slot_bytes=512))
            msg_id = q.enqueue(b"hello world")
            # ... wait for processing ...
            result = q.try_dequeue_result(msg_id)
            q.shutdown()
        
        Or with context manager:
            with GpuQueue(QueueConfig(1024, 512)) as q:
                msg_id = q.enqueue(b"hello")
    )doc")
        .def(py::init<const QueueConfig&>(), py::arg("config") = QueueConfig{},
             "Create a GpuQueue with the given configuration")
        
        // Enqueue: accepts bytes or buffer protocol
        .def("enqueue", [](GpuQueue& q, py::bytes data, uint32_t timeout_ms) {
            std::string s = data;
            int64_t result = q.enqueue(s.data(), s.size(), timeout_ms);
            if (result < 0) {
                throw py::value_error("Enqueue failed: " + std::to_string(result));
            }
            return static_cast<uint64_t>(result);
        }, py::arg("data"), py::arg("timeout_ms") = 1000,
           "Enqueue a message. Returns message ID on success.")
        
        .def("enqueue", [](GpuQueue& q, py::buffer data, uint32_t timeout_ms) {
            py::buffer_info info = data.request();
            int64_t result = q.enqueue(info.ptr, static_cast<size_t>(info.size * info.itemsize), timeout_ms);
            if (result < 0) {
                throw py::value_error("Enqueue failed: " + std::to_string(result));
            }
            return static_cast<uint64_t>(result);
        }, py::arg("data"), py::arg("timeout_ms") = 1000,
           "Enqueue a buffer (numpy array, etc). Returns message ID.")
        
        .def("enqueue_nowait", [](GpuQueue& q, py::bytes data) {
            std::string s = data;
            int64_t result = q.enqueue(s.data(), s.size(), 0);
            if (result < 0) {
                if (result == static_cast<int64_t>(QueueStatus::ERR_FULL)) {
                    throw py::value_error("Queue is full");
                }
                throw py::value_error("Enqueue failed: " + std::to_string(result));
            }
            return static_cast<uint64_t>(result);
        }, py::arg("data"),
           "Enqueue without waiting. Raises if queue is full.")
        
        // Dequeue result
        .def("try_dequeue_result", [](GpuQueue& q, uint64_t msg_id, size_t max_len) {
            std::vector<uint8_t> buffer(max_len);
            size_t actual_len = max_len;
            QueueStatus status = q.try_dequeue_result(msg_id, buffer.data(), &actual_len);
            
            if (status == QueueStatus::OK) {
                buffer.resize(actual_len);
                return py::make_tuple(true, py::bytes(reinterpret_cast<char*>(buffer.data()), actual_len));
            } else if (status == QueueStatus::ERR_NOT_READY) {
                return py::make_tuple(false, py::none());
            } else {
                throw py::value_error("Dequeue failed: " + std::to_string(static_cast<int>(status)));
            }
        }, py::arg("msg_id"), py::arg("max_len") = 4096,
           "Try to dequeue result. Returns (success, data) tuple.")
        
        // Poll completions
        .def("poll_completions", [](GpuQueue& q, size_t max_count) {
            std::vector<uint64_t> msg_ids(max_count);
            size_t count = q.poll_completions(msg_ids.data(), max_count);
            msg_ids.resize(count);
            return msg_ids;
        }, py::arg("max_count") = 64,
           "Poll for completed message IDs. Returns list of IDs.")
        
        // Control
        .def("is_running", &GpuQueue::is_running,
             "Check if the queue is running")
        .def("shutdown", &GpuQueue::shutdown,
             "Shutdown the queue gracefully")
        
        // Stats
        .def("stats", &GpuQueue::stats,
             "Get current queue statistics")
        .def("depth", &GpuQueue::depth,
             "Get current queue depth")
        .def("config", &GpuQueue::config, py::return_value_policy::reference_internal,
             "Get queue configuration")
        
        // Context manager support
        .def("__enter__", [](GpuQueue& q) -> GpuQueue& { return q; })
        .def("__exit__", [](GpuQueue& q, py::object, py::object, py::object) {
            q.shutdown();
        })
        
        .def("__repr__", [](const GpuQueue& q) {
            return std::string("GpuQueue(capacity=") + std::to_string(q.config().capacity) +
                   ", running=" + (q.is_running() ? "True" : "False") +
                   ", depth=" + std::to_string(q.depth()) + ")";
        });
}
