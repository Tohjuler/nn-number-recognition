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

const TRAINING_RATE = 0.05;
const EPOCHS = 10;
const EXPORT_NAME = "full-shuffle-low-lr";
// const BATCH_SIZE = 32;

// ---

async function main() {
	const lr = parseArgNum("lr", TRAINING_RATE);
	const epochs = parseArgNum("epochs", EPOCHS);
	const exportName = parseArg("export", EXPORT_NAME);

	const trainData = await getDataset("train");
	const validationData = await getDataset("test");
	const trainingData = trainData;

	const nn = createNetwork(
		randomNetworkData([784, 128, 64, 10]),
		[sigmoid, sigmoid, identity],
		[sigmoidDerivative, sigmoidDerivative, identityDerivative],
	);
	console.log(`Starting training for ${epochs} epochs...`);
	const res = nn.train(trainingData, lr, epochs, {
		validationData,
		onEpochEnd: (data) => {
			console.log(
				`Epoch ${data.epoch + 1}: Average Loss = ${data.averageLoss}, Time = ${formatTime(data.time)}, Avg Forward Time = ${formatTime(data.averageForwardTime)}${data.validationLoss ? `, Validation Loss = ${data.validationLoss}, Validation Accuracy = ${(data.validationAccuracy! * 100).toFixed(2)}%, Validation Time = ${formatTime(data.validationTime || 0)}` : ""}`,
			);
		},
		lossWithDelta: softmaxCrossEntropyLossWithDelta,
		lossFunction: softmaxCrossEntropyLoss,
		noShuffle: false,
		isOutputLogits: true,
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
		Array.from({ length: epochs }, (_, i) => ({
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
		`./output/training-results-${exportName}-${epochs}.json`,
		JSON.stringify(res, null, 2),
	);
	nn.exportData(`./output/network-${exportName}-${epochs}.json`);
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

function parseArg(name: string, defaultValue: string): string {
	const arg = process.argv.find((arg) => arg.startsWith(`--${name}=`));
	if (arg) {
		return arg.split("=")[1]!;
	}
	return defaultValue;
}

function parseArgNum(name: string, defaultValue: number): number {
	const value = parseArg(name, defaultValue.toString());
	return parseFloat(value);
}