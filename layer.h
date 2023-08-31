#pragma once
#ifndef __MYLAYER__
#define __MYLAYER__
#include "node.h"

class layer 
{
private:
	size_t node_amount;
	vector<node> output_node;
	vector<long double> outputs;
	vector<long double> inputs;
	long double (*activation_function_deriv)(long double);


public:
	layer(size_t nodes, long double (*act_func)(long double), long double (*act_func_deriv)(long double));
	layer(vector<long double> outputs);
	size_t get_node_amount();
	vector<vector<long double>> get_weights();
	vector<long double> get_unactivated();
	vector<long double> get_outputs();
	void compute(vector<long double> inputs);
	void set_outputs(vector<long double> new_outputs);
	long double layer_derivative(long double x);
	void update_weights(vector<vector<long double>> dw, long double alpha);
	void update_biases(vector<long double> db, long double alpha);
};

#endif