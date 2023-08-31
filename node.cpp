#include "node.h"

node::node(long double (*act_func)(long double))
{
	this->inputs = { 0.0 };
	this->inputs_length = inputs.size();
	for (size_t i = 0; i < inputs_length; i++)
		this->weights.push_back(static_cast<double>(rand()) / RAND_MAX);
	this->bias = static_cast<double>(rand()) / RAND_MAX;
	this->weighted_sum = 0.0;
	this->activation_function = act_func;
	// All class vars initalised 
}

long double node::sum()
{
	long double sum = 0.0;
	for (size_t i = 0; i < inputs_length; i++)
	{
		 sum += this->inputs[i]*this->weights[i];
	}
	return (sum + bias);
}

long double node::activate()
{
	return (*this->activation_function)(this->weighted_sum);
}

void node::compute(vector<long double> inputs)
{
	this->inputs = inputs;
	this->weighted_sum = sum();
	this->result = activate();
}

long double node::output()
{
	return result;
}

vector<long double> node::get_weights()
{
	return weights;
}

void node::set_weights(vector<long double> new_weight, long double alpha)
{
	for (size_t i = 0; i < new_weight.size(); i++)
		this->weights[i] = this->weights[i] - new_weight[i] * alpha;
}

void node::set_biases(long double new_bias)
{
	this->bias = this->bias - new_bias;
}

long double node::get_unactivated()
{
	return this->weighted_sum;
}

