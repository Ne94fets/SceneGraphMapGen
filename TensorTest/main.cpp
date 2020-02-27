#include <iostream>

#include <Model.h>
#include <Tensor.h>

#include <memory>

using namespace std;

int main()
{
	cout << "Hello World!" << endl;

	Model m("../model.pb");
	Tensor* tin;
	Tensor* tout;
	return 0;
}
