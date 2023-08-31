#pragma once
# ifndef __UTILS__
#define __UTILS__
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include <iostream>

using namespace std;

vector<vector<long double>> openCSV(string path);
vector<long double> split_mnist(vector<vector<long double>> &vec);
template<typename T>
void pop_front(std::vector<T>& vec)
{
    assert(!vec.empty());
    vec.front() = move(vec.back());
    vec.pop_back();
};
#endif // !__UTILS__
