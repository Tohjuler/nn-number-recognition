import { denormalize, normalize } from "../algorithms/normalization";
import { relu, reluDerivative } from "../algorithms/relu";
import { sigmoid, sigmoidDerivative } from "../algorithms/sigmoid";
import createNetwork, { prepareData, randomNetworkData } from "../network";
import { pipe } from "../utils";

const nn = createNetwork(
	randomNetworkData([2, 4, 1]),
	[relu, sigmoid, sigmoid],
	[reluDerivative, sigmoidDerivative, sigmoidDerivative],
);
// Example usage with house price prediction
const houseData = [
	{ inputs: [1400, 3], expected: [200000] },
	{ inputs: [1600, 3], expected: [230000] },
	{ inputs: [1700, 3], expected: [245000] },
	{ inputs: [1875, 4], expected: [275000] },
	{ inputs: [1100, 2], expected: [180000] },
	{ inputs: [2350, 4], expected: [320000] },
	{ inputs: [2100, 4], expected: [305000] },
	{ inputs: [1500, 3], expected: [215000] },
];

const { normalizedInputs, normalizedTargets, data } = prepareData(houseData);
nn.train(data, 0.1, 10000, undefined, (epoch, averageLoss) => {
	if (epoch % 100 === 0) {
		console.log(`Epoch ${epoch}, Average Loss: ${averageLoss}`);
	}
});

// Create prediction function
const predictHousePrice = pipe<number[]>(
	(house) => [
		...normalize(
			[house[0]!],
			normalizedInputs.min[0]!,
			normalizedInputs.max[0]!,
		),
		...normalize(
			[house[1]!],
			normalizedInputs.min[1]!,
			normalizedInputs.max[1]!,
		),
	],
	nn.forward,
	(prediction) =>
		denormalize(
			prediction,
			normalizedTargets.min[0]!,
			normalizedTargets.max[0]!,
		),
	(res) => res.map((value) => Math.round(value)),
);

// Test the network with new houses
console.log("\nPredictions:");
[
	[1550, 3],
	[2000, 4],
	[1200, 2],
].forEach((house) => {
	const prediction = predictHousePrice(house)[0];
	console.log(
		`House with ${house[0]} sq ft and ${house[1]} bedrooms: $${prediction}`,
	);
});
