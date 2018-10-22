#pragma once
#include <vector>

namespace neuronet
{
	class Neuron;

	typedef std::vector<Neuron> Layer;

	class Net
	{
	public:
		/**
		* 
		*/
		Net(const std::vector<unsigned> &topology); // struct/totpology of neuronnet
		void feedForvard(const std::vector<double> &inputVals); //filup for data
		void backProp(const std::vector<double> &targetVals); // lerning alghorytm
		void getResults(std::vector<double> &resultsVals); //get results
		double getRecentAverageError() const { return recentAverageError; }
	private:
		std::vector<Layer> layers;  // layers[layNum][NeurNum]
		double error;
		double recentAverageError;
		static double recentAverageSmoothingFactor;
	};
}
