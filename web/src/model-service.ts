import { createNetwork, type LayerData, type Network } from "@neural-network/core";
import { identity, identityDerivative } from "@neural-network/core/algorithms/identity";
import { relu, reluDerivative } from "@neural-network/core/algorithms/relu";
import { sigmoid, sigmoidDerivative } from "@neural-network/core/algorithms/sigmoid";
import { softmax } from "@neural-network/core/algorithms/softmax";

type ActivationName = "identity" | "sigmoid" | "relu";

type RuntimeConfig = {
	activations: ActivationName[];
	applySoftmax: boolean;
};

type RawModelConfig = {
	activations?: unknown;
	activation?: unknown;
	hiddenActivation?: unknown;
	outputActivation?: unknown;
	applySoftmax?: unknown;
};

export class NeuralModelService {
	private modelData: LayerData[] | null = null;
	private network: Network | null = null;
	private runtimeConfig: RuntimeConfig | null = null;

	async load(url: string): Promise<LayerData[]> {
		const res = await fetch((import.meta.env.VITE_BASE_URL ?? "") + url);
		if (!res.ok) {
			throw new Error(`Failed to load model: HTTP ${res.status}`);
		}

		const parsed: unknown = await res.json();
		const config = await this.tryLoadConfig(getConfigPathFromModelPath(url));
		return this.loadFromData(parsed, config);
	}

	async loadUploaded(data: unknown, fileName: string): Promise<LayerData[]> {
		const config = await this.tryLoadConfig(getConfigPathFromModelPath(fileName));
		return this.loadFromData(data, config);
	}

	loadFromData(data: unknown, configData?: unknown): LayerData[] {
		if (!isLayerDataArray(data)) {
			throw new Error("Model format is invalid.");
		}

		const config = buildRuntimeConfig(data.length, configData);
		const activations = buildActivationSequence(config.activations);
		const activationDerivatives = buildActivationDerivativeSequence(
			config.activations,
		);

		this.modelData = data;
		this.runtimeConfig = config;
		this.network = createNetwork(data, activations, activationDerivatives);
		return data;
	}

	predict(input: number[]): number[] {
		if (!this.modelData || !this.network) {
			throw new Error("Model not loaded yet.");
		}
		const logits = this.network.forward(input);
		if (this.runtimeConfig?.applySoftmax ?? true) {
			return softmax(logits);
		}
		return logits;
	}

	private async tryLoadConfig(path: string): Promise<unknown | undefined> {
		const res = await fetch((import.meta.env.VITE_BASE_URL ?? "") + path);
		if (!res.ok) {
			return undefined;
		}

		try {
			return await res.json();
		} catch {
			return undefined;
		}
	}
}

function isLayerDataArray(value: unknown): value is LayerData[] {
	if (!Array.isArray(value) || value.length === 0) return false;

	return value.every((layer) => {
		if (typeof layer !== "object" || layer === null || !("neurons" in layer)) {
			return false;
		}

		const neurons = (layer as { neurons?: unknown }).neurons;
		if (!Array.isArray(neurons)) return false;

		return neurons.every((neuron) => {
			if (
				typeof neuron !== "object" ||
				neuron === null ||
				!("weights" in neuron) ||
				!("bias" in neuron)
			) {
				return false;
			}

			const weights = (neuron as { weights?: unknown }).weights;
			const bias = (neuron as { bias?: unknown }).bias;
			return isNumericArray(weights) && typeof bias === "number";
		});
	});
}

function isNumericArray(value: unknown): value is number[] {
	return Array.isArray(value) && value.every((item) => typeof item === "number");
}

function getConfigPathFromModelPath(modelPath: string): string {
	const modelName = modelPath.split("/").pop()?.trim();
	if (!modelName) {
		return "/models-cnf/default.json";
	}
	return `/models-cnf/${modelName}`;
}

function buildRuntimeConfig(layerCount: number, configData?: unknown): RuntimeConfig {
	const defaultActivations = buildDefaultActivationNames(layerCount);
	const defaultConfig: RuntimeConfig = {
		activations: defaultActivations,
		applySoftmax: true,
	};

	if (!isRawModelConfig(configData)) {
		return defaultConfig;
	}

	const activations = parseActivationNames(configData, layerCount) ?? defaultActivations;
	const applySoftmax =
		typeof configData.applySoftmax === "boolean"
			? configData.applySoftmax
			: defaultConfig.applySoftmax;

	return {
		activations,
		applySoftmax,
	};
}

function isRawModelConfig(value: unknown): value is RawModelConfig {
	return typeof value === "object" && value !== null;
}

function parseActivationNames(
	configData: RawModelConfig,
	layerCount: number,
): ActivationName[] | null {
	if (Array.isArray(configData.activations)) {
		const names = configData.activations.filter(isActivationName);
		if (names.length === layerCount) {
			return names;
		}
	}

	if (isActivationName(configData.activation)) {
		if (layerCount === 1) {
			return [configData.activation];
		}
		return [...Array(layerCount - 1).fill(configData.activation), "identity"];
	}

	const hidden = isActivationName(configData.hiddenActivation)
		? configData.hiddenActivation
		: "sigmoid";
	const output = isActivationName(configData.outputActivation)
		? configData.outputActivation
		: "identity";

	if (layerCount === 1) {
		return [output];
	}

	return [...Array(layerCount - 1).fill(hidden), output];
}

function isActivationName(value: unknown): value is ActivationName {
	return value === "identity" || value === "sigmoid" || value === "relu";
}

function buildDefaultActivationNames(count: number): ActivationName[] {
	if (count === 1) {
		return ["identity"];
	}

	return [...Array(count - 1).fill("sigmoid"), "identity"];
}

function buildActivationSequence(names: ActivationName[]) {
	return names.map((name) => {
		switch (name) {
			case "identity":
				return identity;
			case "relu":
				return relu;
			case "sigmoid":
				return sigmoid;
		}
	});
}

function buildActivationDerivativeSequence(names: ActivationName[]) {
	return names.map((name) => {
		switch (name) {
			case "identity":
				return identityDerivative;
			case "relu":
				return reluDerivative;
			case "sigmoid":
				return sigmoidDerivative;
		}
	});
}
