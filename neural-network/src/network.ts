import mse from "./algorithms/mean-square-error";
import { normalize } from "./algorithms/normalization";
import createLayer from "./layer";
import type {
	Activation,
	ActivationDerivative,
	LayerData,
	Network,
    TrainOptions,
} from "./types";

export default function createNetwork(
	layerData: LayerData[],
	activation: Activation | Activation[],
	activationDerivative: ActivationDerivative | ActivationDerivative[],
): Network {
	if (activation.length !== layerData.length && Array.isArray(activation))
		throw new Error("Activation function must be provided for each layer.");
	if (
		activationDerivative.length !== layerData.length &&
		Array.isArray(activationDerivative)
	)
		throw new Error(
			"Activation derivative function must be provided for each layer.",
		);

	const layers = layerData.map(({ neurons }, i) =>
		createLayer(
			neurons,
			Array.isArray(activation) ? activation[i]! : activation,
			Array.isArray(activationDerivative)
				? activationDerivative[i]!
				: activationDerivative,
		),
	);

	const forward = (inputs: number[]): number[] => {
		return layers.reduce(
			(currentInputs, layer) => layer.forward(currentInputs),
			inputs,
		);
	};

	const backward = (target: number[], learningRate: number): void => {
		const lastLayerDeltas = layers[layers.length - 1]!.getNeurons().map(
			(neuron, i) => {
				const state = neuron.getState();
				return (
					(state.lastOutput! - target[i]!) *
					neuron.activationDerivative(state.lastActivation!)
				);
			},
		);

		let nextLayerDeltas = lastLayerDeltas;
		for (let i = layers.length - 1; i >= 0; i--) {
			nextLayerDeltas = layers[i]!.backward(nextLayerDeltas, learningRate);
		}
	};

	const train = (
		trainingData: { inputs: number[]; expected: number[] }[],
		learningRate: number,
		epochs: number,
		options?: TrainOptions,
	): void => {
		const lossFunction = options?.lossFunction || mse;
		for (let epoch = 0; epoch < epochs; epoch++) {
			const start = performance.now();
			const avgForward: number[] = [];
			const avgBackward: number[] = [];
			let loss = 0;
			let i = 0;
			for (const { inputs, expected } of trainingData) {
				const forwardStart = performance.now();
				const output = forward(inputs);
				avgForward.push(performance.now() - forwardStart);
				if ((options?.batchSize && i > 0 && i % options.batchSize === 0) || !options?.batchSize) {
					const startBackward = performance.now();
					backward(expected, learningRate);
					avgBackward.push(performance.now() - startBackward);
				}
				loss += lossFunction(output, expected);

				i++;
			}
			if (options?.onEpochEnd)
				options.onEpochEnd({
					epoch,
					averageLoss: loss / trainingData.length,
					time: performance.now() - start,
					averageForwardTime:
						avgForward.reduce((a, b) => a + b, 0) / avgForward.length,
					averageBackwardTime:
						avgBackward.reduce((a, b) => a + b, 0) / avgBackward.length,
				});
		}
	};

	const getLayers = () => layers;

	// Data func

	const exportData = (filePath: string) => {
		const data = layers.map((layer) => ({
			neurons: layer.getNeurons().map((neuron) => {
				const state = neuron.getState();
				return {
					weights: state.weights,
					bias: state.bias,
				};
			}),
		}));

		// Save to file
		Bun.write(filePath, JSON.stringify(data, null, 2));
	};

	return {
		forward,
		backward,
		train,
		getLayers,
		exportData,
	};
}

export function randomNetworkData(layerSizes: number[]): LayerData[] {
	return layerSizes.slice(1).map((size, i) => ({
		// We skip first layer since it's the input layer
		neurons: Array.from({ length: size }, () => ({
			weights: Array.from(
				{ length: layerSizes[i] || 0 },
				() => Math.random() * 2 - 1,
			),
			bias: Math.random() * 2 - 1,
		})),
	}));
}

// Data should be [[inputs], ]
export function prepareData(data: { inputs: number[]; expected: number[] }[]): {
	normalizedInputs: { data: number[][]; min: number[]; max: number[] };
	normalizedTargets: { data: number[][]; min: number[]; max: number[] };
	data: { inputs: number[]; expected: number[] }[];
} {
	const inputRows: number[][] = Array.from(
		{ length: data[0]!.inputs.length },
		(_, i) => data.map((d) => d.inputs[i]!),
	);
	const targetRows: number[][] = Array.from(
		{ length: data[0]!.expected.length },
		(_, i) => data.map((d) => d.expected[i]!),
	);

	const minMaxInputs = inputRows.map((row) => ({
		min: Math.min(...row),
		max: Math.max(...row),
	}));

	const minMaxTargets = targetRows.map((row) => ({
		min: Math.min(...row),
		max: Math.max(...row),
	}));

	const normalizedInputRows = inputRows.map((row, i) =>
		normalize(row, minMaxInputs[i]!.min, minMaxInputs[i]!.max),
	);

	const normalizedTargetRows = targetRows.map((row, i) =>
		normalize(row, minMaxTargets[i]!.min, minMaxTargets[i]!.max),
	);

	const normalizedInputs = {
		data: normalizedInputRows[0]!.map((_, i) => [
			...normalizedInputRows.map((row) => row[i]!), // We take the i-th element from each normalized input row to form the input vector for the i-th data point
		]),
		min: minMaxInputs.map((m) => m.min),
		max: minMaxInputs.map((m) => m.max),
	};

	const normalizedTargets = {
		data: normalizedTargetRows[0]!.map((_, i) => [
			...normalizedTargetRows.map((row) => row[i]!), // We take the i-th element from each normalized target row to form the target vector for the i-th data point
		]),
		min: minMaxTargets.map((m) => m.min),
		max: minMaxTargets.map((m) => m.max),
	};

	return {
		normalizedInputs,
		normalizedTargets,
		data: normalizedInputs.data.map((inputs, i) => ({
			inputs,
			expected: normalizedTargets.data[i]!,
		})),
	};
}
