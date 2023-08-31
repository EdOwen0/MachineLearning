#pragma once
#ifndef __MYNODE__
#define __MYNODE__
#include <cassert>
#include "MyMaths.h"

class node 
{
private:
	long double weighted_sum;
	long double bias;
	long double result;
	size_t inputs_length;
	vector<long double> inputs;
	vector<long double> weights;
	long double (*activation_function)(long double);

public:
	node(long double (*act_func)(long double));
	long double sum();
	long double activate();
	void compute(vector<long double> inputs);
	long double output();
	vector<long double> get_weights();
	void set_weights(vector<long double> new_weight, long double alpha);
	void set_biases(long double new_bias);
	long double get_unactivated();
};

#endif