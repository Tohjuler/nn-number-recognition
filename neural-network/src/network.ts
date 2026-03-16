import mse from "./algorithms/mean-square-error";
import { normalize } from "./algorithms/normalization";
import { softmax } from "./algorithms/softmax";
import createLayer from "./layer";
import type {
	Activation,
	ActivationDerivative,
	LayerData,
	Network,
	TrainOptions,
	TrainResult,
} from "./types";
import { shuffle, timerHelper } from "./utils";

declare const Bun:
	| {
			write(filePath: string, data: string): unknown;
	  }
	| undefined;

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

	const backward = (
		target: number[],
		learningRate: number,
		outputDeltaZ?: number[],
	): void => {
		let deltaZ = outputDeltaZ; // If using softmax+CE, then it is give, else we compute it now.

		if (!deltaZ) {
			// Fallback, assumes we use MSE loss func.
			// deltaZ_i = (a_i - y_i) * f'(z_i)
			const lastLayer = layers[layers.length - 1]!;
			deltaZ = lastLayer.getNeurons().map((neuron, i) => {
				const state = neuron.getState();
				if (state.lastActivation === null || state.lastOutput === null)
					throw new Error(
						"Neuron state is missing activation or output for backward pass.",
					);
				return (
					(state.lastOutput - target[i]!) *
					neuron.activationDerivative(state.lastActivation)
				);
			});
		}

		// Backdrop
		for (let idx = layers.length - 1; idx >= 0; idx--) {
			const layer = layers[idx]!;
			const deltaAprev = layer.backward(deltaZ, learningRate);

			// Prepare deltaZ for prev (next) layer
			const prevLayer = layers[idx - 1];
			if (!prevLayer) break; // No more layers to propagate back to

			// deltaZ_prev = deltaA_prev '* f'(z_prev)
			deltaZ = prevLayer.getNeurons().map((neuron, i) => {
				const state = neuron.getState();
				if (state.lastActivation === null)
					throw new Error(
						"Neuron state is missing activation for backward pass.",
					);
				return (
					deltaAprev[i]! * neuron.activationDerivative(state.lastActivation)
				);
			});
		}
	};

	const train = (
		trainingData: { inputs: number[]; expected: number[] }[],
		learningRate: number,
		epochs: number,
		options?: TrainOptions,
	): TrainResult => {
		const lossFunction = options?.lossFunction || mse;
		const lossOverEpochs: number[] = [];
		const validationLossOverEpochs: number[] = [];
		const validationAccuracyOverEpochs: number[] = [];
		const times = timerHelper();

		times.start("totalTraining");
		for (let epoch = 0; epoch < epochs; epoch++) {
			times.start("epoch");

			let loss = 0;
			for (const { inputs, expected } of options?.noShuffle ? trainingData : shuffle(trainingData)) {
				times.start("forward");
				const output = forward(inputs);
				times.end("forward", "averageForward");

				if (options?.lossWithDelta) {
					times.start("lossWithDelta");
					const res = options.lossWithDelta(output, expected);
					times.end("lossWithDelta", "averageLossWithDelta");

					loss += res.loss;
					times.start("backward");
					backward(expected, learningRate, res.deltaZ);
					times.end("backward", "averageBackward");
				} else {
					times.start("loss");
					loss += lossFunction(output, expected);
					times.end("loss", "averageLoss");

					times.start("backward");
					backward(expected, learningRate);
					times.end("backward", "averageBackward");
				}
			}
			const avgLoss = loss / trainingData.length;
			lossOverEpochs.push(avgLoss);
			const epochTime = times.end("epoch", "averageEpoch");

			let validationLoss: number | null = null;
			let validationAccuracy: number | null = null;
			let validationTime: number | null = null;
			if (options?.validationData) {
				times.start("validation");
				let correct = 0;
				validationLoss =
					options.validationData.reduce((sum, { inputs, expected }) => {
						const output = forward(inputs);
						const probs = options?.isOutputLogits ? softmax(output) : output;
						const predicted = probs.indexOf(Math.max(...probs));
						const actual = expected.indexOf(Math.max(...expected));
						if (predicted === actual) correct++;
						return (
							sum +
							(options?.lossWithDelta
								? options.lossWithDelta(output, expected).loss
								: lossFunction(output, expected))
						);
					}, 0) / options.validationData.length;
				validationAccuracy = correct / options.validationData.length;

				validationTime = times.end("validation", "averageValidation");
				validationLossOverEpochs.push(validationLoss);
				validationAccuracyOverEpochs.push(validationAccuracy);
			}

			if (options?.onEpochEnd)
				options.onEpochEnd({
					epoch,
					averageLoss: avgLoss,
					time: epochTime,
					averageForwardTime: times.getAvg("averageForward") || 0,
					validationLoss: validationLoss || undefined,
					validationTime: validationTime || undefined,
					validationAccuracy: validationAccuracy || undefined,
				});
		}
		times.endAndSave("totalTraining");

		return {
			epoch: epochs,
			averageLoss: lossOverEpochs[lossOverEpochs.length - 1]!,
			lossOverEpochs,
			validationLoss:
				validationLossOverEpochs.length > 0
					? validationLossOverEpochs[validationLossOverEpochs.length - 1]!
					: undefined,
			validationLossOverEpochs:
				validationLossOverEpochs.length > 0
					? validationLossOverEpochs
					: undefined,
			validationAccuracy:
				validationAccuracyOverEpochs.length > 0
					? validationAccuracyOverEpochs[
							validationAccuracyOverEpochs.length - 1
						]!
					: undefined,
			validationAccuracyOverEpochs:
				validationAccuracyOverEpochs.length > 0
					? validationAccuracyOverEpochs
					: undefined,
			times: times.getOutput(),
		};
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

		if (!Bun) {
			throw new Error("exportData is only available in Bun runtime.");
		}

		Bun.write(filePath, JSON.stringify(data, null, 2));
	};

	return {
		forward,
		backward,
		train,
		getLayers,
		exportData,
	} satisfies Network;
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
