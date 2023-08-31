#include "model.h"



model::model()
{
	
	/*
	* vector<vector<double>> x_train training questions
	* vector<vector<double>> x_test  training answers
	* double				 alpha   learning rate
	* 
	*/
	this->alpha = 0.1;
}

model::model(long double alpha)
{
	/*
	* vector<vector<double>> x_train training questions
	* vector<vector<double>> x_test  training answers
	* double				 alpha   learning rate
	*
	*/
	assert(alpha);
	this->alpha = alpha;
}

void model::add_layer(size_t nodes, string function)
{
	assert(nodes);
	size_t last = this->layers.size() - 1;
	if (function == "ReLu")
		this->layers.push_back(layer(nodes, &Mymaths::ReLu, &Mymaths::ReLu_derivative));
	if (function == "LReLu")
		this->layers.push_back(layer(nodes, &Mymaths::LReLu, &Mymaths::LReLu_derivative));
	if (function == "sigmoid")
		this->layers.push_back(layer(nodes, &Mymaths::sigmoid, &Mymaths::sigmoid_derivative));
}

void model::forwardprop()
{
	this->layer_outputs.resize(this->layers.size());
	
	for (size_t i = 1; i < x_test.size(); i++)
	{
		this->layers[0].set_outputs(this->x_test[i]);
		for (size_t currentlayer = 1; currentlayer < this->layers.size(); currentlayer++) 
		{
			this->layers[currentlayer].compute(this->layers[currentlayer - 1].get_outputs());
			this->layer_outputs[currentlayer].push_back(this->layers[currentlayer].get_outputs());
		}
	}
}

void model::backprop_update(vector<long double> y, vector<vector<long double>> output)
{
	size_t m = x_test.size();
	
	// dZ and sizing
	vector<vector<vector<long double>>> dZ;
	dZ.resize(this->layers.size());

	// dW and sizing
	vector<vector<vector<long double>>> dW;
	dW.resize(this->layers.size());

	// db and sizing
	vector<vector<long double>> db;
	db.resize(this->layers.size());

	// backprop for output layer
	size_t lastLayerIndex = this->layers.size() - 1;
	size_t weights_size = this->layers[lastLayerIndex - 1].get_weights().size();
	vector<long double> previous_layer_outputs = this->layers[lastLayerIndex - 1].get_outputs();

	for (size_t currentquestion = 0; currentquestion < output.size(); currentquestion++) 
	{
		dZ[lastLayerIndex].resize(output.size());
		for( size_t j = 0; j < y.size(); j++)
			dZ[lastLayerIndex][currentquestion].push_back(output[currentquestion][j] - y[j]);
	
		
		dW[lastLayerIndex].resize(output[0].size());
		/*
		* layer_outputs		[4, 9999, [784,8,8,10]]
		* dZ				[4, 9999, [784,8,8,10]]
		* dW				[4, [8,8, 10], [784][8][8] ] ]
		* 
		*/

		for (size_t currentweight = 0; currentweight < output[0].size(); currentweight++) 
		{
			dW[lastLayerIndex][currentweight].push_back( 1.0 / m * (dZ[lastLayerIndex][currentquestion][currentweight] * this->layer_outputs[lastLayerIndex - 1][currentquestion][currentweight]));
		}
		db[lastLayerIndex].push_back(1.0 / m * Mymaths::sum(dZ[lastLayerIndex][currentquestion]));
	}
	




	
	/*
	// Backprop for hidden layers
	for (size_t currentLayerIndex = lastLayerIndex - 1; currentLayerIndex >= 1; currentLayerIndex--)
	{
		for (size_t i = 0; i < this->layers[currentLayerIndex].get_outputs().size(); i++) 
		{
			vector<long double> previous_layer_outputs = this->layers[currentLayerIndex - 1].get_outputs();

			this->layers[currentLayerIndex].layer_derivative(previous_layer_outputs[i]);
			vector<vector<long double>> front_weights = this->layers[currentLayerIndex + 1].get_weights();
			vector<vector<long double>> compiled_weights;
			for (size_t j = 0; j < front_weights.size(); j++) 
				compiled_weights.push_back(Mymaths::dot(dZ[currentLayerIndex + 1], front_weights[j]));
			//dZ[currentLayerIndex].push_back(Mymaths::dot());
		}
		db[currentLayerIndex].push_back(1.0 / m * Mymaths::sum(dZ[currentLayerIndex]));
	}*/

	for (size_t l = 0; l < this->layers.size() - 1; l++)
	{
		this->layers[l].update_weights(dW[l], this->alpha);
		this->layers[l].update_biases(db[l], this->alpha);
	}


}

void model::train(vector<vector<long double>> x_train, vector<long double> x_test, size_t epochs)
{
	this->layers.insert(this->layers.begin(), layer(x_train[0]));
	this->x_test = Mymaths::one_hot(x_test);
	
	
	
	if (epochs <= 0)
		throw invalid_argument("Number of epochs must be greater than 0");
	if (this->layers[this->layers.size()-1].get_node_amount() != this->x_test[0].size())
		throw invalid_argument("Final layer cannot be unequal to training targets");

	
	
	
	for (size_t i = 0; i < epochs; i++) {
		this->forwardprop();
		cout << "Is this loss?  " << Mymaths::cross_entropy(this->layer_outputs[this->layer_outputs.size()-1], this->x_test[i]) << "\n";
		backprop_update(this->x_test[i], this->layer_outputs[this->layer_outputs.size()-1]);
		
		this->layer_outputs.clear();
	}
}

vector<long double> model::get_final()
{
	size_t last = this->layers.size() - 1;
	for (size_t i = 0; i < this->layers.size(); i++)
		layers[i].get_outputs();
	return this->layers[last].get_outputs();
}

size_t model::get_model_length()
{
	return this->layers.size();
}
