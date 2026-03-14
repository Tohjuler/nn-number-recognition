import createNeuron from "./neuron";
import type { Activation, ActivationDerivative, Layer, NeuronData } from "./types";

export default function createLayer(
	neuronsData: NeuronData[],
	activation: Activation,
	activationDerivative: ActivationDerivative,
): Layer {
	const neurons = neuronsData.map(({ weights, bias }) =>
		createNeuron(weights, bias, activation, activationDerivative),
	);

	const forward = (inputs: number[]): number[] => {
		return neurons.map((neuron) => neuron.activate(inputs));
	};

	/**
	 * backward(deltaZ, lr) expects deltaZ per neuron:
	 * deltaZ[i] = dLoss/dZ_i  (Z = pre-activation)
	 *
	 * Returns deltaA_prev (dLoss/dA_prev), where A_prev are the inputs to this layer.
	 */
	const backward = (
		deltaZ: number[],
		learningRate: number,
	): number[] => {
		const oldWeights = neurons.map((n) => n.getState().weights.slice());

		for (let i = 0; i < neurons.length; i++) {
			neurons[i]!.updateWeights(learningRate, deltaZ[i]!); // Update the weights
		}

		// Compute the deltaA_prev
		const prevSize = oldWeights[0]?.length ?? 0; // Number of inputs to this layer
		if (prevSize === 0) return []; // No inputs, so no deltas to propagate back

		const deltaAprev = new Array(prevSize).fill(0);
		for (let i = 0; i < prevSize; i++) {
			let sum = 0;
			for (let j = 0; j < oldWeights.length; j++) {
				sum += oldWeights[j]![i]! * deltaZ[j]!;
			}
			deltaAprev[i] = sum;
		}

		return deltaAprev;
	};

	const getNeurons = () => neurons;

	return {
		forward,
		backward,
		getNeurons,
	};
}
