#include "net.h"
#include "neuron.h"

using namespace neuronet;

Net::Net(const std::vector<unsigned> &topology)
{
	unsigned numLayers = topology.size();

	for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
	{
		layers.push_back(Layer());
		unsigned numOutputs =
			layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

		for (unsigned neuroNum = 0; neuroNum <= topology[layerNum]; ++neuroNum)
		{
			layers.back().push_back(Neuron(numOutputs, neuroNum));
		}

		layers.back().back().setOutputVal(1.0);
	}
}

void Net::feedForvard(const std::vector<double> &inputVals)
{
	assert(inputVals.size() == layers[0].size() - 1);

	for (unsigned i = 0; i < inputVals.size(); ++i)
	{
		layers[0][i].setOutputVal(inputVals[i]);
	}

	for (unsigned layerNum = 1; layerNum < layers.size() ; ++layerNum) 
	{
		Layer &prevLayer = layers[layerNum - 1];
		for (unsigned n = 0; n < layers[layerNum].size() - 1; ++n)
		{
			layers[layerNum][n].feedForward(prevLayer);
		}
	}
}

void Net::backProp(const std::vector<double> &targetVals)
{
	Layer &outputLayer = layers.back();
	error = 0.0;

	for ( unsigned n = 0; n < outputLayer.size(); ++n) 
	{
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		error += delta * delta;
	}
	error /= outputLayer.size() - 1;
	error = sqrt(error);

	recentAverageError =
		(recentAverageError * recentAverageSmoothingFactor + error)
		/ (recentAverageSmoothingFactor + 1.0);

	for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		outputLayer[n].clacOutputGradients(targetVals[n]);
	}

	for (unsigned layersNum = 0; )
}