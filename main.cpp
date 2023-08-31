#include "MyMaths.h"
#include "node.h"
#include "layer.h"
#include "model.h"
#include "util.h"

int main()
{

	// known questions
	vector<vector<long double>> x_train = openCSV("dataset//mnist_test.csv");
	// known answers
	vector<long double> x_test = split_mnist(x_train);
	
	model m = model();
	m.add_layer(8, "ReLu");
	m.add_layer(8, "LReLu");
	m.add_layer(10, "sigmoid");
	
	m.train(x_train , x_test, 10);
	vector<long double> kk = m.get_final();
	
	for (int i = 0; i < kk.size(); i++)
		cout << kk[i] << " ";
	return 0;
}