#pragma once
#ifndef __MYMODEL__
#define __MYMODEL__
#include <string.h>
#include "layer.h"
#include "encode.h"

class model 
{
private:
	long double alpha;
	vector<layer> layers;
	vector<vector<long double>> x_test;
	vector<vector<vector<long double>>> layer_outputs;
public:
	model();
	model(long double alpha);
	void add_layer(size_t nodes, string function);
	void forwardprop();
	void backprop_update(vector<long double> y, vector<vector<long double>> output);
	void train(vector<vector<long double>> x_train, vector<long double> x_test, size_t epochs);
	vector<long double> get_final();
	// returns number of layers
	size_t get_model_length();
};
#endif