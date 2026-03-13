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

	const backward = (
		nextLayerDeltas: number[],
		learningRate: number,
	): number[] => {
		const deltas = neurons.map((neuron, i) => {
			const neuronState = neuron.getState();
			if (neuronState.lastActivation === null)
				throw new Error(
					"No activation stored for neuron during backward pass.",
				);
			const oldWeights = neuronState.weights;
			const delta =
				nextLayerDeltas[i]! *
				neuron.activationDerivative(neuronState.lastActivation);
			neuron.updateWeights(learningRate, delta);
			return oldWeights.map((weight) => weight * delta);
		});
        const firstNeuronDelta = deltas[0];
        if (!firstNeuronDelta) throw new Error("No deltas calculated for backward pass.");

		return firstNeuronDelta.map((_, inputIndex) =>
			deltas.reduce((sum, deltas) => sum + deltas[inputIndex]!, 0), // We sum all the deltas for each weight
		);
	};

	const getNeurons = () => neurons;

	return {
		forward,
		backward,
		getNeurons,
	};
}
