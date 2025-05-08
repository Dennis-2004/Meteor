#pragma once

#include <queue>
#include <mutex>
#include <vector>
#include <string>
#include "globals.h"  // for myType

// A single client request: party's share of input + metadata
struct ClientRequest {
    std::string client_id;
    int request_id;
    MEVectorType input_data;
    int socket_fd;
};

class RequestQueue {
public:
    void addRequest(const std::string& client_id, int request_id, const MEVectorType& input_data, int socket_fd);
    std::vector<ClientRequest> getNextBatch(size_t batchSize);
    size_t size();

private:
    std::queue<ClientRequest> queue_;
    std::mutex mutex_;
};

extern RequestQueue requestQueue;