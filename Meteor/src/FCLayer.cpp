
#pragma once
#include "FCLayer.h"
#include "Functionalities.h"
#include "secondary.h"
#include "connect.h"
using namespace std;

FCLayer::FCLayer(FCConfig *conf, int _layerNum)
	: Layer(_layerNum),
	  conf(conf->inputDim, conf->batchSize, conf->outputDim),
	  activations(conf->batchSize * conf->outputDim),
	  deltas(conf->batchSize * conf->outputDim),
	  weights(conf->inputDim * conf->outputDim),
	  biases(conf->outputDim)
{
	initialize();
}

void FCLayer::initialize()
{
	// Initialize weights and biases here.
	// Ensure that initialization is correctly done.
	size_t lower = 30;
	size_t higher = 50;
	size_t decimation = 10000;
	size_t size = weights.size();

	float temp[size];
    for (size_t i = 0; i < size; ++i)
    {
        temp[i] = (float)(rand() % (higher - lower) + lower) / decimation;
        myType val = floatToMyType(temp[i]);

        // All shares initialized for Meteor as RSS component only, public part = 0
        if (partyNum == PARTY_A)
            weights[i] = make_pair(0, make_pair(val, 0));
        else if (partyNum == PARTY_B)
            weights[i] = make_pair(0, make_pair(0, 0));
        else if (partyNum == PARTY_C)
            weights[i] = make_pair(0, make_pair(0, val));
    }

	fill(biases.begin(), biases.end(), make_pair(0, make_pair(0, 0)));
}

void FCLayer::printLayer()
{
	cout << "----------------------------------------------" << endl;
	cout << "(" << layerNum + 1 << ") FC Layer\t\t  " << conf.inputDim << " x " << conf.outputDim << endl
		 << "\t\t\t  "
		 << conf.batchSize << "\t\t (Batch Size)" << endl;
}

void FCLayer::forward(const MEVectorType &inputActivation)
{
	log_print("FC.forward");
	// cout << "FC.forward" << endl;
	size_t rows = conf.batchSize;
	size_t columns = conf.outputDim;
	size_t common_dim = conf.inputDim;
	size_t size = rows * columns;
	// start_communication();
	if (FUNCTION_TIME)
		cout << "Meteor_funcMatMul: " << funcTime(Meteor_funcMatMul, inputActivation, weights, activations, rows, common_dim, columns, 0, 0, FLOAT_PRECISION) << " milliseconds" << endl;
	else
		Meteor_funcMatMul(inputActivation, weights, activations, rows, common_dim, columns, 0, 0, FLOAT_PRECISION);

	for (size_t r = 0; r < rows; ++r)
	{
		for (size_t c = 0; c < columns; ++c)
		{
			activations[r * columns + c].first = activations[r * columns + c].first + biases[c].first;
			activations[r * columns + c].second = activations[r * columns + c].second + biases[c].second;
		}
	}
	// end_m("fc");
	// pause_communication();
	// end_communication("FC" + to_string(layerNum));
	// resume_communication();
	// start_m();
}

void FCLayer::computeDelta(MEVectorType &prevDelta)
{
	log_print("FC.computeDelta");
	/*
		//Back Propagate
		size_t rows = conf.batchSize;
		size_t columns = conf.inputDim;
		size_t common_dim = conf.outputDim;

		if (FUNCTION_TIME)
			cout << "funcMatMul: " << funcTime(funcMatMul, deltas, weights, prevDelta, rows, common_dim, columns, 0, 1, FLOAT_PRECISION) << endl;
		else
			funcMatMul(deltas, weights, prevDelta, rows, common_dim, columns, 0, 1, FLOAT_PRECISION);
			*/
}

void FCLayer::updateEquations(const MEVectorType &prevActivations)
{
	log_print("FC.updateEquations");
	/*
		size_t rows = conf.batchSize;
		size_t columns = conf.outputDim;
		size_t common_dim = conf.inputDim;
		size_t size = rows*columns;
		RSSVectorMyType temp(columns, std::make_pair(0,0));

		//Update Biases
		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < columns; ++j)
				temp[j] = temp[j] + deltas[i*columns + j];

		funcTruncate(temp, LOG_MINI_BATCH + LOG_LEARNING_RATE, columns);
		subtractVectors<RSSMyType>(biases, temp, biases, columns);

		//Update Weights
		rows = conf.inputDim;
		columns = conf.outputDim;
		common_dim = conf.batchSize;
		size = rows*columns;
		RSSVectorMyType deltaWeight(size);

		if (FUNCTION_TIME)
			cout << "funcMatMul: " << funcTime(funcMatMul, prevActivations, deltas, deltaWeight, rows, common_dim, columns, 1, 0, FLOAT_PRECISION + LOG_LEARNING_RATE + LOG_MINI_BATCH) << endl;
		else
			funcMatMul(prevActivations, deltas, deltaWeight, rows, common_dim, columns, 1, 0,
						FLOAT_PRECISION + LOG_LEARNING_RATE + LOG_MINI_BATCH);

		subtractVectors<RSSMyType>(weights, deltaWeight, weights, size);
		*/
}
