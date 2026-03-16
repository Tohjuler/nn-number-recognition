import { identity, identityDerivative } from "@neural-network/core/algorithms/identity";
import { sigmoid, sigmoidDerivative } from "@neural-network/core/algorithms/sigmoid";
import { softmax } from "@neural-network/core/algorithms/softmax";
import { createNetwork, type LayerData, type Network } from "neural-network";

export class NeuralModelService {
	private modelData: LayerData[] | null = null;
	private network: Network | null = null;

	async load(url: string): Promise<LayerData[]> {
		const res = await fetch(url);
		if (!res.ok) {
			throw new Error(`Failed to load model: HTTP ${res.status}`);
		}

		const parsed: unknown = await res.json();
		if (!isLayerDataArray(parsed)) {
			throw new Error("Model format is invalid.");
		}

		this.modelData = parsed;
		this.network = createNetwork(
			parsed,
			[sigmoid, sigmoid, identity],
			[sigmoidDerivative, sigmoidDerivative, identityDerivative],
		);
		return parsed;
	}

	predict(input: number[]): number[] {
		if (!this.modelData || !this.network) {
			throw new Error("Model not loaded yet.");
		}
		const logits = this.network.forward(input);
		return softmax(logits);
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
