package main

import "github.com/stuntgoat/nn_utils"


// An element in a hidden layer that takes values from
// an array of neurons, sums
type Neuron struct {
	// a reference to the neurons that send
	// values to this neuron from the previous layer
	// or the inputs
	inputs []*Neuron

	// the values that this node received from each neuron or input from the previous layer
	// in the network
	inputValues []float64

	// a reference to the Neurons in the next layer or the
	// output
	outputs []*Neuron

	// the array of weights from this Neuron
	// to each of the Neurons or output in the next
	// layer
	outputWeights []float64
}


// activationFunction calls the sigmoid function on the sum and
// func (neuron *Neuron) activationFunction(sum, weight float64) result float64 {
// }


// updateNextLayer caluclates the values the next layer in the neural network or outputs.
// For each neuron in the outputs, send the sum of the values received from the inputs
// called with the sigmoid function; this result is multiplied by corresponding output weights
func (neuron *Neuron) calculateNextLayer() result []float64 {
	var sum = 0
	for _, val := range inputValues {
		sum += val
	}

	for index, currentValue := range neuron.ouputs {
		nextValue = nn_utils.Sigmoid_float(sum * neuron.ouputWeights[index])
		append(result, nextValue)
	}
	return
}


//
type Output struct {
	inputs []*Neuron

	// values last received from inputs
	inputValues []float64
}

// result returns the sum of the *output's inputValues and returns the
// sigmoid function called with the sum
func (ouput Output) result() float64 {
	var sum = 0.0
	for _, val := range inputValues {
		sum += val
	}
	return nn_utils.Sigmoid_float(value)
}

type NeuralNetLayer struct {
	neurons []Neurons
}

// allWeights returns a single array of all of the ouputWeights
// for each neuron in the `layer`.
func (layer *NeuralNetLayer) allWeights() weights []float64 {
	for _, neuron := layer.neurons {
		weights = append(weights, neuron.ouputWeights)
	}
	return
}

type NeuralNetConfiguration struct {
	numInputs int
	numOutputs int

	numHiddenLayers int
	numNeuronsPerLayer int
}

//
func associateLayerWithPrevious(previousLayer, layer NeuralNetLayer) {

}

// createNeuralNet is a method that returns a NeuralNet type from the values
// in `config`.
func (config *NeuralNetConfiguration) createNeuralNet() neuralNet NeuralNet {
	for i := 0; i < config.numInputs; i++ {
		neuron := Neuron{}
		neuralNet.inputs = append(neuralNet.inputs, neuron)
	}
	var currentLayer = *NeuralNetLayer
	var previousLayer = *NeuralNetLayer

	for i := 0; i < config.numHiddenLayers; i++ {

		layer := NeuralNetLayer{}
		currentLayer = &layer

		for j := 0; j < config.numNeuronsPerLayer; j++ {
			neuron := Neuron{}
			layer.neurons = append(layer.neurons, neuron)
		}

		if i == 0 {
			currentLayer, previousLayer = associateLayerWithPrevious(&neuralNet.inputs, currentLayer)
		} else {
			associateLayerWithNeurons(previousLayer, currentLayer)
		}
		previousLayer = currentLayer
	}
	return
}


// NeuralNet contains the logic and data for representing a prediction model.
type NeuralNet struct {

	inputsLayer NeuralNetLayer

	// all of the hidden layers
	layers []NeuralNetLayer

	outputs []Outputs
}


// predict takes an array of floats and returns the output values
// after passing them through the neural network.
func (neuralNet *NeuralNet) predict(theta []float64) predicted []float64 {
	return
}

// getWeights iterates through the neural network and returns the
// sequence of output weights for each node(input, neuron, or output) in the network.
// These weights are used in gradient descent for training the network
func (neuralNet *NeuralNet) getAllWeights() weights []float64 {
	for _, input := range neuralNet.inputs {
		weights = append(weights, input.outputWeights)
	}
	for _, layer := range neuralNet.layers {
		layerWeights := layer.allWeights()
		for _, weight := layerWeights {
			weights = append(weights, weight)
		}
	}
	return
}

// setAllWeights sets the weights on each input, and neuron within the NeuralNet.
func (neuralNet *NeuralNet) setAllWeights(weights []float64) {
	weightIndex = 0
	for _, input := range neuralNet.inputs {
		for outIndex, _ := range input.outputs {
			input.outputWeights[outIndex] = weights[weightIndex]
			weightIndex += 1
		}
	}
	for _, layer := range neuralNet.layers {
		for _, neuron := layer.neuron {
			for outIndex, _ := neuron.ouputs {
				neuron.outputWeights[outIndex] = weights[weightIndex]
				weightIndex += 1
			}
		}
	}
}

// returns the error between each weight and the outputs
func (neuralNet *NeuralNet) getError(predicted, expected []float64) error []float64 {
	return
}

// type NeuralNetConstructor struct {
// 	numInputs int
// 	numOutputs int
// 	numHiddenLayers int
// 	numNeuronsPerLayer int
// }

// // createInputs creates Input objects, the first hidden layer of Neurons and
// // connects them with eachother.
// func (constructor *NeuralNetConstructor) createNetwork() neural_network NeuralNetwork {

// 	for i := 0; i < constructor.numInputs; i++ {

// 	}
// }



// // ConstructNeuralNet returns an instance of a NeuralNetwork.
// func ConstructNeuralNet(numInputs, numOutputs, numHiddenLayers, numNeuronsPerLayer int) neural_net NeuralNet {

// }

	// // maps input events to their weights
	// weights map[string]float64

	// // this maps an event string to a neuron in the first
	// // hidden layer.
	// input_vector map[string]*Neuron
