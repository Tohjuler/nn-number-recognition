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
	averageLoss: number;
	time: number;
	averageForwardTime: number;
	averageBackwardTime: number;
}

export type TrainOptions = {
	batchSize?: number;
	onEpochEnd?: (data: EpochData) => void;
	lossFunction?: (output: number[], expected: number[]) => number;
}

export interface Network {
	forward(inputs: number[]): number[];
	backward(target: number[], learningRate: number): void;
	train(
		trainingData: { inputs: number[]; expected: number[] }[],
		learningRate: number,
		epochs: number,
		options?: TrainOptions,
	): void;
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
