#include "neuron.h"

using namespace neuronet;

double Neuron::eta = 0.15; // net learning rate
double Neuron::alpha = 0.5; // momentum

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
	for ( unsigned c = 0; c < numOutputs; ++c)
	{
		outputWeights.push_back(Connection());
		outputWeights.back().weight = randomWeight();
	}

	m_myIndex = myIndex;
}

void Neuron::updateInputWeights(Layer &prevLayer)
{
	for (unsigned n = 0; n < prevLayer.size(); ++n)
	{
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.outputWeights[m_myIndex].deltaWeight;

		double newDeltaWeight = 
			eta * neuron.getOutputVal() * gradient + alpha + oldDeltaWeight;
		neuron.outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.outputWeights[m_myIndex].weight += newDeltaWeight;

	}
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
	double sum = 0.0;
	for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
	{
		sum += outputWeights[n].weight * nextLayer[n].gradient;
	}

	return sum;
}

void Neuron::clacHiddenGradients(const Layer &nextLayer)
{
	double dow = sumDOW(nextLayer);
	gradient = dow * Neuron::activationFunctionDerivative(outputVal);
}

void Neuron::clacOutputGradients(double targetsVals)
{
	double delta = targetsVals - outputVal;
	gradient = delta * Neuron::activationFunctionDerivative(outputVal);
}

double Neuron::activationFunction( double x )
{
	return tanh(x);
}

double Neuron::activationFunctionDerivative(double x)
{
	return 1.0 - x * x;
}

void Neuron::feedForward(const Layer &prevLayer)
{
	double sum = 0.0;

	for (unsigned n = 0; n < prevLayer.size() ; ++n)
	{
		sum += prevLayer[n].getOutputVal() * 
			prevLayer[n].outputWeights[m_myIndex].weight;
	}

	outputVal = Neuron::activationFunction(sum);
}