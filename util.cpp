#include "util.h"

vector<vector<long double>> openCSV(string path)
{
    cout << "Loading Dataset";
    ifstream fin(path);
    vector<vector<long double>> file;
    vector<long double> file_line;
    string temp, line, word;
    while (getline(fin, temp)) 
    {
        string w = "";
        for (auto x : temp)
        {
            if (x == ',')
            {
                file_line.push_back(stoi(w));
                w = "";
            }
            else {
                w += x;
            }
        }
        file.push_back(file_line);
        file_line.clear();
    }
    fin.close();
    cout << " Complete" << endl;
    return file;
}

vector<long double> split_mnist(vector<vector<long double>> &vec)
{
    vector<long double> x;
    for (size_t i = 0; i < vec.size(); i++) 
    {
        x.push_back(vec[i][0]);
        pop_front(vec[i]);
    }
        
    return x;
}
