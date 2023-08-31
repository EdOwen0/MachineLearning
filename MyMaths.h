#pragma once
#ifndef __MYMATHS__
#define __MYMATHS__
// #include <cstdlib>
// #include <ctime>
// #include <algorithm>
#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

namespace Mymaths 
{
	// ReLu
	long double ReLu(long double x);
	long double ReLu_derivative(long double x);
	
	//softmax
	void softmax(double* input, double* output, unsigned int n);

	// sigmoid
	long double sigmoid(long double x);
	long double sigmoid_derivative(long double x);
	
	// Leaky ReLu
	long double LReLu(long double x);
	long double LReLu_derivative(long double x);
	
	// LOSS / COST FUNCTION
	 long double cross_entropy(vector<vector<long double>> output, vector<long double> target);

	// Summation
	long double sum(vector<long double> x);
	// Product
	long double product(vector<long double> x);
	// Dot Product
	vector<long double> dot(vector<long double> x, vector<long double> y);
}
#endif