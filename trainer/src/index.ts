import { createNetwork, randomNetworkData } from "neural-network";
import { sigmoid, sigmoidDerivative } from "neural-network/algorithms/sigmoid";
import getDataset from "./mnist-data";

// Vars
// ---

const TRAINING_RATE = 0.1;
const EPOCHS = 10;
const BATCH_SIZE = 32;

// ---

async function main() {
	const trainData = await getDataset("train");

	const nn = createNetwork(
		randomNetworkData([784, 128, 64, 10]),
		sigmoid,
		sigmoidDerivative,
	);
	const start = performance.now();
	console.log(`Starting training for ${EPOCHS} epochs...`);
	nn.train(trainData, TRAINING_RATE, EPOCHS, {
		batchSize: BATCH_SIZE,
		onEpochEnd: (data) => {
			console.log(
				`Epoch ${data.epoch + 1}: Average Loss = ${data.averageLoss}, Time = ${formatTime(data.time)}, Avg Forward Time = ${formatTime(data.averageForwardTime)}, Avg Backward Time = ${formatTime(data.averageBackwardTime)}`,
			);
		},
	});
	console.log(
		`Training completed in ${formatTime(performance.now() - start)} for ${EPOCHS} epochs.`,
	);

	nn.exportData(`./output/network-${EPOCHS}-${BATCH_SIZE}.json`);
}
main();

// ---

function formatTime(ms: number): string {
	const seconds = Math.floor(ms / 1000);
	const minutes = Math.floor(seconds / 60);
	const hours = Math.floor(minutes / 60);

	if (hours > 0) {
		return `${hours}h ${minutes % 60}m ${seconds % 60}s`;
	}
	if (minutes > 0) {
		return `${minutes}m ${seconds % 60}s`;
	}
	if (seconds > 0) {
		return `${seconds}s ${Math.round(ms % 1000)}ms`;
	}

	return `${ms}ms`;
}
