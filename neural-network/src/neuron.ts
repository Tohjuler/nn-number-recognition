import type {
	Activation,
	ActivationDerivative,
	Neuron,
	NeuronState,
} from "./types";

export default function createNeuron(
	weights: number[],
	bias: number,
	activation: Activation,
	activationDerivative: ActivationDerivative,
): Neuron {
	const neuronState: NeuronState = {
		weights,
		bias,
		lastInputs: null,
		lastActivation: null,
		lastOutput: null,
	};

	const activate = (inputs: number[]): number => {
		if (inputs.length !== neuronState.weights.length)
			throw new Error("Input length must match weights length.");
		const weightedSum =
			inputs.reduce(
				(sum, input, i) => sum + input * neuronState.weights[i]!,
				0,
			) + neuronState.bias;
        
		neuronState.lastInputs = inputs;
		neuronState.lastActivation = weightedSum;
		neuronState.lastOutput = activation(weightedSum);
		return neuronState.lastOutput;
	};

	const updateWeights = (learningRate: number, delta: number): void => {
		neuronState.weights = neuronState.weights.map(
			(weight, i) => weight - learningRate * delta * (neuronState.lastInputs?.[i] || 0),
		);
		neuronState.bias -= learningRate * delta;
	};

	const getState = (): NeuronState => ({
		...neuronState,
	});

	return {
		activate,
		updateWeights,
		getState,
		activationDerivative: activationDerivative,
	};
}
