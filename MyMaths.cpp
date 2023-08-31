#include "MyMaths.h"






long double Mymaths::ReLu(long double x)
{
	if (x <= 0) return 0.0;
	return x;
}

long double Mymaths::ReLu_derivative(long double x)
{
	if (x <= 0) return 0.0;
	return 1.0;
}

/*

double input[3] = {3.0, 4.0, 5.0};
size_t size = sizeof input / sizeof input[0]

double output[size];


Mymaths::softmax(input,output, size);

*/
void Mymaths::softmax(double* input, double* output, unsigned int n)
{
	double sum = 0;

	double m = -INFINITY;
	for (unsigned int i = 0; i < n; i++)
	{
		m = max(m, input[i]);
	}

	for (unsigned int j = 0; j < n; j++)
	{
		sum += exp(input[j] - m);
	}

	for (unsigned int i = 0; i < n; i++)
	{
		output[i] = exp(input[i] - m) / sum;
	}
}

long double Mymaths::sigmoid(long double x)
{
	// x/(1+|x|) sigmoid approx
	// 1 / expl(-x) + 1 has precision loss and returns 1
	return x / (1.0 + fabs(x));
}

long double Mymaths::sigmoid_derivative(long double x)
{
	return x * (1.0 - x);
}

long double Mymaths::LReLu(long double x)
{
	if (x < 0) return x * 0.01;
	return x;
}

long double Mymaths::LReLu_derivative(long double x)
{
	if (x < 0) return 0.01;
	return x;
}

/*
##################################################
see entropy_test()
#############################################
*/
long double Mymaths::cross_entropy(vector<vector<long double>> output, vector<long double> target)
{
	long double loss = 0;
	for (size_t j = 0; j < output.size(); j++)
		for (size_t i = 0; i < target.size(); i++)
			loss += target[i]*log(output[j][i]);
	return fabs(loss);
}

long double Mymaths::sum(vector<long double> x)
{
	long double sum = 0.0;
	for (size_t i = 0; i < x.size(); i++)
		sum += x[i];
	return sum;
}


long double Mymaths::product(vector<long double> x)
{
	long double product = 0.0;
	for (size_t i = 0; i < x.size(); i++)
		product *= x[i];
	return product;
}

vector<long double> Mymaths::dot(vector<long double> x, vector<long double> y)
{
	
	if (x.size() != y.size())
		throw invalid_argument("Vector sizes must be equal.");

	vector<long double> result;
	for (size_t i = 0; i < x.size(); ++i)
		result.push_back(x[i] * y[i]);
	return result;
}
