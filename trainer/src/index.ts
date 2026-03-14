import { createNetwork, randomNetworkData } from "neural-network";
import {
	softmaxCrossEntropyLoss,
	softmaxCrossEntropyLossWithDelta,
} from "neural-network/algorithms/cross-entropy-loss";
import {
	identity,
	identityDerivative,
} from "neural-network/algorithms/identity";
import { sigmoid, sigmoidDerivative } from "neural-network/algorithms/sigmoid";
import getDataset from "./mnist-data";

// Vars
// ---

const TRAINING_RATE = 0.1;
const EPOCHS = 10;
const EXPORT_NAME = "full";
// const BATCH_SIZE = 32;

// ---

async function main() {
	const trainData = await getDataset("train");
	const validationData = await getDataset("test");
	const trainingData = trainData;

	const nn = createNetwork(
		randomNetworkData([784, 128, 64, 10]),
		[sigmoid, sigmoid, identity],
		[sigmoidDerivative, sigmoidDerivative, identityDerivative],
	);
	console.log(`Starting training for ${EPOCHS} epochs...`);
	const res = nn.train(trainingData, TRAINING_RATE, EPOCHS, {
		validationData,
		onEpochEnd: (data) => {
			console.log(
				`Epoch ${data.epoch + 1}: Average Loss = ${data.averageLoss}, Time = ${formatTime(data.time)}, Avg Forward Time = ${formatTime(data.averageForwardTime)}${data.validationLoss ? `, Validation Loss = ${data.validationLoss}, Validation Accuracy = ${(data.validationAccuracy! * 100).toFixed(2)}%, Validation Time = ${formatTime(data.validationTime || 0)}` : ""}`,
			);
		},
		lossWithDelta: softmaxCrossEntropyLossWithDelta,
		lossFunction: softmaxCrossEntropyLoss,
	});
	console.log(
		`Training completed in ${formatTime(res.times.totalTraining || 0)}. Final Average Loss: ${res.averageLoss}${res.validationLoss ? `, Final Validation Loss: ${res.validationLoss}, Final Validation Accuracy: ${(res.validationAccuracy! * 100).toFixed(2)}%` : ""}`,
	);

	console.log("\nTime breakdown:");
	console.table(
		Object.entries(res.times).map(([key, value]) => ({
			Operation: key,
			Time: formatTime(value),
		})),
	);

	console.log("\nFinal Metrics:");
	console.table(
		Array.from({ length: EPOCHS }, (_, i) => ({
			Epoch: i + 1,
			"Average Loss": res.lossOverEpochs[i]!,
			"Validation Loss": res.validationLossOverEpochs
				? res.validationLossOverEpochs[i]
				: "N/A",
			"Validation Accuracy": res.validationAccuracyOverEpochs
				? `${(res.validationAccuracyOverEpochs[i]! * 100).toFixed(2)}%`
				: "N/A",
		})),
	);

	// Save the trained model

	// nn.exportData(`./output/network-${EPOCHS}-${BATCH_SIZE}.json`);
	Bun.write(
		`./output/training-results-${EXPORT_NAME}-${EPOCHS}.json`,
		JSON.stringify(res, null, 2),
	);
	nn.exportData(`./output/network-${EXPORT_NAME}-${EPOCHS}.json`);
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

	return `${ms.toFixed(6)}ms`;
}
