export type NeuronState = {
	weights: number[];
	bias: number;
	lastInputs: number[] | null;
	lastActivation: number | null;
	lastOutput: number | null;
};

export interface Neuron {
	activate(inputs: number[]): number;
	updateWeights(learningRate: number, delta: number): void;
	activationDerivative(x: number): number;
	getState(): NeuronState;
}

export interface Layer {
	forward(inputs: number[]): number[];
	backward(nextLayerDeltas: number[], learningRate: number): number[];
	getNeurons(): Neuron[];
}

export type EpochData = {
	epoch: number;
	time: number;
	averageLoss: number;
	validationLoss?: number;
	validationAccuracy?: number;
	validationTime?: number;
	averageForwardTime: number;
}

export type LossResult = {
	loss: number;
	deltaZ?: number[]; // Used for output layer (dLoss/dZ_out)
}

export type TrainOptions = {
	validationData?: { inputs: number[]; expected: number[] }[];
	// batchSize?: number;
	onEpochEnd?: (data: EpochData) => void;

	// Loss
	lossFunction?: (output: number[], expected: number[]) => number;
	lossWithDelta?: (output: number[], expected: number[]) => LossResult;

	noShuffle?: boolean;
	isOutputLogits?: boolean; // If true, the loss functions will apply softmax internally
}

export type TrainResult = {
	epoch: number;
	averageLoss: number;
	lossOverEpochs: number[];
	validationLoss?: number;
	validationLossOverEpochs?: number[];
	validationAccuracy?: number;
	validationAccuracyOverEpochs?: number[];
	times: Record<string, number>;
}

export interface Network {
	forward(inputs: number[]): number[];
	backward(target: number[], learningRate: number, outputDeltaZ?: number[]): void;
	train(
		trainingData: { inputs: number[]; expected: number[] }[],
		learningRate: number,
		epochs: number,
		options?: TrainOptions,
	): TrainResult;
	getLayers(): Layer[];
	exportData(filename: string): void;
}

export type Activation = (x: number) => number;
export type ActivationDerivative = (x: number) => number;

// Data Types
// ---

export type NeuronData = {
	weights: number[];
	bias: number;
};

export type LayerData = {
	neurons: NeuronData[];
};
