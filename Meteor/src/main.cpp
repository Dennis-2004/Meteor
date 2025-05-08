#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "AESObject.h"
#include "Precompute.h"
#include "secondary.h"
#include "connect.h"
#include "NeuralNetConfig.h"
#include "NeuralNetwork.h"
#include "unitTests.h"
#include "RequestQueue.h"

int partyNum;
AESObject *aes_indep;
AESObject *aes_next;
AESObject *aes_prev;
Precompute PrecomputeObject;
RequestQueue requestQueue;

// Synchronization primitives
std::mutex mtx;
std::condition_variable cv;

int main(int argc, char **argv)
{
	/****************************** PREPROCESSING ******************************/
	parseInputs(argc, argv);
	NeuralNetConfig *config = new NeuralNetConfig(NUM_ITERATIONS);
	string network, dataset, security;
	bool PRELOADING = false;

	if (argc == 9)
	{
		network = argv[6];
		dataset = argv[7];
		security = argv[8];
	}
	else
	{
		network = "SecureML";
		dataset = "MNIST";
		security = "Semi-honest";
	}

	selectNetwork(network, dataset, security, config);
	config->checkNetwork();
	NeuralNetwork *net = new NeuralNetwork(config);

	aes_indep = new AESObject(argv[3]);
	aes_next = new AESObject(argv[4]);
	aes_prev = new AESObject(argv[5]);

	initializeCommunication(argv[2], partyNum);
	synchronize(2000000);

	/****************************** INFERENCE ******************************/
	network += " preloaded";
	PRELOADING = true;
	preload_network(PRELOADING, network, net);

	start_m();

	std::thread listener([&]()
						 { listenForRequests(5000 + partyNum); });
	listener.detach();

	std::cout << "[P" << partyNum << "] Listening for client input..." << std::endl;

	int num_sleep = 0;

	// Loop forever to check for batches and perform inference
	while (true)
	{
		size_t queue_size = requestQueue.size();
		if (queue_size >= MINI_BATCH_SIZE || (num_sleep > 10 && queue_size > 0))
		{
			num_sleep = 0;
			vector<ClientRequest> batch = requestQueue.getNextBatch(queue_size);
			net->inputData = inputBatch(batch);
			;

			std::cout << "[P" << partyNum << "] Running inference on batch...\n";
			test(PRELOADING, network + " test", net, (int)queue_size);
			returnOutput(batch, net);
		}
		else
		{
			// If no batch, sleep a bit before checking again
			num_sleep++;
			// cout << num_sleep << " " << MINI_BATCH_SIZE << endl;
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
	}

	/****************************** CLEAN-UP ******************************/
	delete aes_indep;
	delete aes_next;
	delete aes_prev;
	delete config;
	delete net;
	deleteObjects();

	return 0;
}
