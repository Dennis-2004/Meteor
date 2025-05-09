#include "RequestQueue.h"
#include <thread>
#include <chrono>
#include <iostream>

void RequestQueue::addRequest(const std::string& client_id, int request_id, const MEVectorType& input_data, int socket_fd) {
    ClientRequest req{client_id, request_id, input_data, socket_fd};

    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(req);
}

std::vector<ClientRequest> RequestQueue::getNextBatch(size_t batchSize) {
    std::vector<ClientRequest> batch;

    std::unique_lock<std::mutex> lock(mutex_);
    while (queue_.size() < batchSize) {
        lock.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        lock.lock();
    }

    for (size_t i = 0; i < batchSize; ++i) {
        batch.push_back(queue_.front());
        queue_.pop();
    }

    return batch;
}

size_t RequestQueue::size() {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

void RequestQueue::requeueFront(const std::vector<ClientRequest>& requests) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::queue<ClientRequest> new_queue;

    // Add new requests to the front
    for (const auto& req : requests) {
        new_queue.push(req);
    }

    // Append existing queue
    while (!queue_.empty()) {
        new_queue.push(queue_.front());
        queue_.pop();
    }

    queue_ = std::move(new_queue);
}
