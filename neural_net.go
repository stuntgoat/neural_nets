package main

import (
	"fmt"
	"github.com/stuntgoat/nn_utils"
	"math"
	"strconv"

)


type Neuron struct {
	// a reference to the neurons that send
	// values to this neuron from the previous layer
	inputs []*Neuron
	// the array of weights multiplied by the each of the respective inputs
	inputWeights []float64

	// value sent to each node in the next layer
	outputValue float64

	// calculated by back propagation
	delta float64
}

func (neuron *Neuron) String() string {
	return "outputValue: " + strconv.FormatFloat(neuron.outputValue, 'E', -1, 64) + " delta: " + strconv.FormatFloat(neuron.delta, 'E', -1, 64)
}


// For each neuron in the outputs, send the sum of the values received from the inputs
// called with the sigmoid function; this result is multiplied by corresponding output weights
func (neuron *Neuron) calculateOutput() {
	if len(neuron.inputs) == 0 {
		return
	}

	sum := 0.0
	for index, inputNeuron := range neuron.inputs {
		sum += inputNeuron.outputValue * neuron.inputWeights[index]
	}

	value := nn_utils.Sigmoid_float(sum)
	neuron.outputValue = value
}


type NeuralNetLayer struct {
	neurons []*Neuron
}


type NeuralNetConfiguration struct {
	// TODO: - use an array [<input>, <layer1>, . . ., <layerN>, <output>]
	//         to determine both the number of neurons per layer and number
	//         of layers
	numInputs int
	numOutputs int

	numHiddenLayers int
	numNeuronsPerLayer int
}


// NeuralNet contains the logic and data for representing a prediction model.
type NeuralNet struct {

    inputLayer NeuralNetLayer

    // all of the hidden layers
    hiddenLayers []*NeuralNetLayer

    outputLayer NeuralNetLayer

	convergenceThreshold float64

	guess []float64
}


// createNeuralNet is a method that returns a NeuralNet type from the values
// in `config`.
func (config *NeuralNetConfiguration) createNeuralNet() (neuralNet NeuralNet) {
	inputNeuronArray := make([]*Neuron, config.numInputs)
    neuralNet.inputLayer = NeuralNetLayer{inputNeuronArray}

	// create input layer
    for i := 0; i < config.numInputs; i++ {
        neuron := &Neuron{}
        neuralNet.inputLayer.neurons[i] = neuron
    }

    var previousLayer *NeuralNetLayer
    previousLayer = &neuralNet.inputLayer

	// create hidden layers
	neuralNet.hiddenLayers = make([]*NeuralNetLayer, config.numHiddenLayers)
    for i := 0; i < config.numHiddenLayers; i++ {

		layerNeuronArray := make([]*Neuron, config.numNeuronsPerLayer)
        layer := NeuralNetLayer{layerNeuronArray}

        for j := 0; j < config.numNeuronsPerLayer; j++ {

			// if first hidden layer, use num inputs as num of inputWeights
			var numInputs int
			if i == 0 {
				numInputs = config.numInputs
			} else {  // else we use the number of neurons per hidden layer
				numInputs = config.numNeuronsPerLayer
			}
            neuron := &Neuron{
			// TODO: - name fields
				make([]*Neuron, numInputs),
				make([]float64, numInputs),
				0.0,
				0.0,
			}

            layer.neurons[j] = neuron
            for previousIndex, previousNeuron := range previousLayer.neurons {
                neuron.inputs[previousIndex] = previousNeuron
            }
        }
        previousLayer = &layer
        neuralNet.hiddenLayers[i] = &layer
    }

	// create output
	outputNeuronArray := make([]*Neuron, config.numOutputs)
    neuralNet.outputLayer = NeuralNetLayer{outputNeuronArray}

    for i := 0; i < config.numOutputs; i++ {
		neuron := &Neuron{
			make([]*Neuron, config.numNeuronsPerLayer),
			make([]float64, config.numNeuronsPerLayer),
			0.0,
			0.0,
		}
        neuralNet.outputLayer.neurons[i] = neuron
        for previousIndex, previousNeuron := range previousLayer.neurons {
            neuron.inputs[previousIndex] = previousNeuron
        }
    }

	// set weights


    return
}


// predict takes an array of floats and returns the output values
// after passing them through the neural network.
func (neuralNet *NeuralNet) predict(inputs []float64) (predicted []float64) {
    // set inputLayer's outputValues from `inputs`
    for index, value := range inputs {
        neuralNet.inputLayer.neurons[index].outputValue = value
    }

    // iterate over each layer in neuralNet.hiddenLayers and call 'calculateOutput' in each neuron
    for _, layer := range neuralNet.hiddenLayers {
        for _, neuron := range layer.neurons {
            neuron.calculateOutput()
        }
    }

    // call 'calculateOutput' in each neuron of the neuralNet.ouputLayer
	for _, neuron := range neuralNet.outputLayer.neurons {
        neuron.calculateOutput()
        predicted = append(predicted, neuron.outputValue)
    }

    // return an array of the neuron.outputValue in each neuron in the neuralNet.outputLayer
    return
}

// func (neuralNet *NeuralNet) predict(theta []float64) predicted []float64 {
// 	return
// }


//     predictedValues := neuralNet.predict(inputs) // side effect of setting all outputValues in the NeuralNet


//     //PRECONDITION: neuron.delta = 0.0 for all neurons in the network
//     //neuralNet.setDeltasToZero()

//      // loop over output neurons, and set their deltas
//     for index, outputNeuron := range neuralNet.outputLayer.neurons {
//         outputNeuron.delta = outputNeuron.outputValue - expectedOutputs[index]

//         /// also increment the delta for the inputs to outputNeuron

//         for neuronIndex, neuron := range outputNeuron.inputs {
//             neuron.delta += outputNeuron.inputWeights[neuronIndex] * outputNeuron.delta
//         }
//     }
// 	//a   x
// 	//b   y
// 	//c   z
// 	//delta_a = 0 + w(a->x) * delta(x)
// 	//detla_b = 0 + w(b->x) * delta(x)
// 	//delta_c = 0 + w(c->x) * delta(x)
// 	//delta_a = w(a->x) * delta(x) + w(a -> y) * delta(y) + w(a->z) * delta(z)
//     // loop over hidden layers (from output to input)
//     //   in each hidden layer, loop over neurons, and set the deltas of their input neurons
//     // don't do it for the first hidden layer (index 0), because the inputLayer shouldn't have a delta (the delta of input layer doesn't affect training)
//     for i := len(neuralNet.hiddenLayers) - 1; i > 0; i-- {
//         layer = neuralNet.hiddenLayers[i]
//         for neuronIndex, currentNeuron := range layer.neurons {// <<<<< neuron.delta is set. you need to set the deltas of the inputs to
//             for previousIndex, previousNeuron := range currentNeuron.inputs {
//                 previousNeuron.delta += currentNeuron.inputWeights[previousIndex] * currentNeuron.delta
//             }
//         }
//     }
// }


// getWeights iterates through the neural network and returns the
// sequence of output weights for each node(input, neuron, or output) in the network.
// These weights are used in gradient descent for training the network
func (neuralNet *NeuralNet) getAllWeights()(weights []float64) {

	// add weights from hidden layers
	for _, layer := range neuralNet.hiddenLayers {
		for _, neuron := range layer.neurons {
			for _, weight := range neuron.inputWeights {
				weights = append(weights, weight)
			}
		}
	}

	// add weights from output layer
	for _, neuron := range neuralNet.outputLayer.neurons {
		for _, weight := range neuron.inputWeights {
			weights = append(weights, weight)
		}
	}
	return
}

// setAllWeights sets the weights on each input, and neuron within the NeuralNet.
func (neuralNet *NeuralNet) setAllWeights(weights []float64) {
	var weightIndex = 0

	// add new weight to each of the inputsWeights arrays for each
	// neuron in the hidden layers
	for _, layer := range neuralNet.hiddenLayers {
		for _, currentNeuron := range layer.neurons {
 			for inputIndex, _ := range currentNeuron.inputs {
				currentNeuron.inputWeights[inputIndex] = weights[weightIndex]
				weightIndex += 1
			}
		}
	}

	// add the new weights to the inputWeights for each neuron in the
	// output layers
	for _, neuron := range neuralNet.outputLayer.neurons {
		for inputIndex, _ := range neuron.inputs {
			neuron.inputWeights[inputIndex] = weights[weightIndex]
			weightIndex += 1
		}
	}
}


// returns the error between each weight and the outputs
func (neuralNet *NeuralNet) getError(predicted, expected []float64) (error []float64) {
	return
}

func (neuralNet *NeuralNet) getInputs() (inputs []float64) {
	for _, inputNeuron := range neuralNet.inputLayer.neurons {
		inputs = append(inputs, inputNeuron.outputValue)
	}
	return
}

func (neuralNet *NeuralNet) getOutputs() (outputs []float64) {
	for _, outputNeuron := range neuralNet.outputLayer.neurons {
		outputs = append(outputs, outputNeuron.outputValue)
	}
	return
}


func (neuralNet *NeuralNet) clearDeltas() {
	for _, inputNeuron := range neuralNet.inputLayer.neurons {
		inputNeuron.delta = 0.0
	}

	// add weights from hidden layers
	for _, layer := range neuralNet.hiddenLayers {
		for _, neuron := range layer.neurons {
			neuron.delta = 0.0
		}
	}

	// add weights from output layer
	for _, neuron := range neuralNet.outputLayer.neurons {
		neuron.delta = 0.0
	}
}


func (neuralNet *NeuralNet) doBackPropagation(inputs, expectedOutputs []float64) {
    neuralNet.predict(inputs) // side effect of setting all outputValues in the NeuralNet

    //PRECONDITION: neuron.delta = 0.0 for all neurons in the network
    //neuralNet.setDeltasToZero()
	neuralNet.clearDeltas()

     // loop over output neurons, and set their deltas
    for index, outputNeuron := range neuralNet.outputLayer.neurons {
		// fmt.Println("outputNeuron", outputNeuron)
		// fmt.Println("expectedOutputs[index]", expectedOutputs[index])
        outputNeuron.delta = outputNeuron.outputValue - expectedOutputs[index]

		// fmt.Println("outputNeuron", outputNeuron)
        /// also increment the delta for the inputs to outputNeuron
        for neuronIndex, neuron := range outputNeuron.inputs {
            neuron.delta += outputNeuron.inputWeights[neuronIndex] * outputNeuron.delta
        }
    }
//a   x
//b   y
//c   z
//delta_a = 0 + w(a->x) * delta(x)
//detla_b = 0 + w(b->x) * delta(x)
//delta_c = 0 + w(c->x) * delta(x)
//delta_a = w(a->x) * delta(x) + w(a -> y) * delta(y) + w(a->z) * delta(z)


    // loop over hidden layers (from output to input)
    //   in each hidden layer, loop over neurons, and set the deltas of their input neurons
    // don't do it for the first hidden layer (index 0), because the inputLayer shouldn't have a delta (the delta of input layer doesn't affect training)
    for i := len(neuralNet.hiddenLayers) - 1; i > 0; i-- {
        layer := neuralNet.hiddenLayers[i]
        for _, currentNeuron := range layer.neurons {// <<<<< neuron.delta is set. you need to set the deltas of the inputs to
            for previousIndex, previousNeuron := range currentNeuron.inputs {
                previousNeuron.delta += currentNeuron.inputWeights[previousIndex] * currentNeuron.delta
            }
        }
    }
     //delta = (predicted_output - expected_output)  <-- for output layer
     //delta^l = sum_k w_jk delta_k^{l+1}   <-- for hidden layers
     ///  X -> Y1,  X->Y2, X->Y3
     // delta_X = w_(X->Y1) * delta_Y1 +
     //           w_(X->Y2) * delta_Y2 +
     //           w_(X->Y3) * delta_Y3
     /// input layer does not have any delta
}


// TODO: - implement DifferentiatedFunction interface for NeuralNet
func (neuralNet *NeuralNet) f(theta []float64) (sum float64) {
	neuralNet.setAllWeights(theta)

	neuralNet.doBackPropagation(neuralNet.getInputs(), neuralNet.guess)

	for _, outputNeuron := range neuralNet.outputLayer.neurons {
		// fmt.Println("outputNeuron", outputNeuron)
		sum += math.Pow(outputNeuron.delta, 2.0)
	}
	return
}

func (neuralNet *NeuralNet) diff_of_f(theta []float64) (gradient []float64) {
	// self.nn.setAllWeights(theta)  <<<
	neuralNet.setAllWeights(theta)

	// neuralNet.predict(neuralNet.getInputs()) // << sets new outputs from theta as new weights

	// self.nn.doBackPropagation(self.input, self.output)
	neuralNet.doBackPropagation(neuralNet.getInputs(), neuralNet.guess)

	//gradient = [0.0 for i in range(len(theta))]

	//each index in theta corresponds to a weight between two nodes, say X and Y ... X->Y

	// for all X->Y in the network:
	for _, layer := range neuralNet.hiddenLayers {
		for _, currentNeuron := range layer.neurons {
			for _, previousNeuron := range currentNeuron.inputs {
				gradient = append(gradient, previousNeuron.outputValue * currentNeuron.delta)
			}
		}
	}

	for _, currentNeuron := range neuralNet.outputLayer.neurons {
		for _, previousNeuron := range currentNeuron.inputs {
			gradient = append(gradient, previousNeuron.outputValue * currentNeuron.delta)
		}
	}

	// // gradient[X->Y] = X.output * Y.delta
	return
}




func main() {
	config := NeuralNetConfiguration{
		numInputs: 2,
		numOutputs: 2,
		numHiddenLayers: 2,
		numNeuronsPerLayer: 2,
	}

	neuralNet := config.createNeuralNet()
	fmt.Println("neural net created", neuralNet)

	// // TEST: insure that initial weights are 0.0
	// currentWeights := neuralNet.getAllWeights()

	// // TEST: insure that we have the correct number of weights
	// numWeights := len(currentWeights)

	// newWeights := make([]float64, numWeights)

	// vals := []float64{1, 0}

	// TEST: insure sane value with each weight @ 0.0
	// neuralNet.predict(vals)


	// TEST: insure sane value with known weights and inputs
	// for i := 0; i < numWeights; i++ {
	// 	newWeights[i] = .1 * float64(i)
	// }

	// TEST: insure that setAllWeights works
	// neuralNet.setAllWeights(newWeights)

	// currentWeights = neuralNet.getAllWeights()
	// fmt.Println("currentWeights", currentWeights)

	// TEST: insure sane predictions from known weights and vals
	// neuralNet.predict(vals)
	// fmt.Println("newResult", newResult)

	// TEST: insure doBackPropagation sets deltas correctly
	// neuralNet.doBackPropagation(vals, []float64{.1, .9})

	// TEST: insure f returns sane value with known theta and weights
	// f := neuralNet.f(newWeights)

	// fmt.Println("f", f)

	// for i := 0; i < numWeights; i++ {
	// 	newWeights[i] = .2 * float64(i)
	// }

	// vals = neuralNet.diff_of_f(newWeights)
	// fmt.Println("vals", vals)

	input := []float64{1.0, 0.0}
	expected := []float64{.1, .9}

	predicted := neuralNet.predict(input)
	fmt.Println("predicted", predicted)
	fmt.Println("expected", expected)

	shitty_weights := neuralNet.getAllWeights()
	fmt.Println("shitty_weights", shitty_weights)

	numWeights := len(shitty_weights)
	newWeights := make([]float64, numWeights)
	for i := 0; i < numWeights; i++ {
		newWeights[i] = .2 * float64(i)
	}


	neuralNet.guess = expected

	// call back prob with expected
	// neuralNet.doBackPropagation(input, expected)



	new_weights := Gradient_descent(newWeights, .55, neuralNet)
	neuralNet.setAllWeights(new_weights)
	success := neuralNet.predict(input)

	fmt.Println("success", success)
	fmt.Println("expected", expected)
}

// Multiplies `alpha` by every value in `vector2`, then
// adds that result with `vector`
func Scale_add(vector1 []float64, vector2 []float64, alpha float64) (scaled []float64) {
	for i, value := range vector2 {
		scaled = append(scaled, value * alpha + vector1[i])
	}
	return
}

func (neuralNet *NeuralNet) is_converged(vec1, vec2 []float64) bool {
	if neuralNet.convergenceThreshold <= 0.0 {
		neuralNet.convergenceThreshold = float64(1e-8)
	}
	return math.Abs(neuralNet.f(vec1) - neuralNet.f(vec2)) < neuralNet.convergenceThreshold
}


func Gradient_descent(guess_vector []float64, alpha float64, diff_func NeuralNet) (outputs []float64)  {
	last_guess := guess_vector

	outputs = Scale_add(last_guess, diff_func.diff_of_f(last_guess), -alpha)
	fmt.Println("outputs", outputs)
	for {
		if diff_func.is_converged(last_guess, outputs) {
			break
		}

		last_guess = outputs
		outputs = Scale_add(last_guess, diff_func.diff_of_f(last_guess), -alpha)
	}
	return
}
