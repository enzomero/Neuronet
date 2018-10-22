#pragma once
#include <vector>

namespace neuronet
{
	class Neuron;

	typedef std::vector<Neuron> Layer;

	struct Connection
	{
		double weight;
		double deltaWeight;
	};

	class Neuron
	{
	public:
		Neuron(unsigned numOutputs, unsigned myIndex);
		void setOutputVal(double val) { outputVal = val; }
		double getOutputVal() const { return outputVal; }
		void feedForward(const Layer &prevLayer);
		void clacOutputGradients(double targetVals);
		void clacHiddenGradients(const Layer &nextLayer);
		void updateInputWeights(Layer &prevLayers);
	private:
		static double eta;
		static double alpha;
		static double randomWeight() { return rand() / double(RAND_MAX); }
		static double activationFunction(double x);
		static double activationFunctionDerivative(double x);
		double sumDOW(const Layer &nextLayer) const;
		double outputVal;
		std::vector<Connection> outputWeights;
		unsigned m_myIndex;
		double gradient;
	};
}


