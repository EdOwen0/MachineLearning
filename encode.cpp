#include "encode.h"

vector<vector<long double>> Mymaths::one_hot(vector<long double> x)
{
    vector<vector<long double>> y;
    for (size_t i = 0; i < x.size(); i++)
    {
        vector<long double> w;
        for (size_t index = 0; index < 10; index++)
        {
            if (index == x[i])
            {
                w.push_back(1.0);
            }
            else 
            {
                w.push_back(0.0);
            }
            
        }
        y.push_back(w);
    }
    return y;
}
