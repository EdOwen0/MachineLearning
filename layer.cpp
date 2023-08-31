#include "layer.h"

layer::layer(size_t nodes, long double (*act_func)(long double), long double (*act_func_deriv)(long double))
{
	if (nodes <= 0)
		throw invalid_argument("Number of nodes must be greater than 0");
	
	assert(act_func != nullptr);
	assert(act_func_deriv != nullptr);

	this->node_amount = nodes;
	for (size_t i = 0; i < nodes; i++)
		this->output_node.push_back(node(act_func));
	
	this->inputs = { 0.0 };

	this->activation_function_deriv = act_func_deriv;

}

layer::layer(vector<long double> outputs)
{
	this->node_amount = 0;
	this->outputs = outputs;
}

size_t layer::get_node_amount()
{
	return this->node_amount;
}

vector<vector<long double>> layer::get_weights()
{
	vector<vector<long double>> weights;
	weights.reserve(output_node.size());

	for (auto& node : output_node) 
		weights.push_back(node.get_weights());

	return weights;
}

vector<long double> layer::get_unactivated()
{
	vector<long double> unactivated;
	unactivated.reserve(output_node.size());

	for (auto& node : output_node)
		unactivated.push_back(node.get_unactivated());

	return unactivated;
}

vector<long double> layer::get_outputs()
{
	if (outputs.empty() && !output_node.empty()) {
		outputs.reserve(output_node.size());

		for (auto& node : output_node)
			outputs.push_back(node.output());
	}

	return outputs;
}

void layer::compute(vector<long double> inputs)
{
	for (auto& node : output_node)
		node.compute(inputs);
}

void layer::set_outputs(vector<long double> new_outputs)
{

}

long double layer::layer_derivative(long double x)
{
	return (*this->activation_function_deriv)(x);
}

void layer::update_weights(vector<vector<long double>> dw, long double alpha)
{
	for (size_t i = 0; i < node_amount; i++)
		output_node[i].set_weights(dw[i], alpha);	
}

void layer::update_biases(vector<long double> db, long double alpha)
{
	if (db.size() <= 0)
		throw invalid_argument("Difference of bias must be greater than 0");


	for (size_t i = 0; i < node_amount; i++)
		output_node[i].set_biases(db[i]*alpha);
}
