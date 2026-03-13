import { relu, reluDerivative } from "../algorithms/relu";
import { sigmoid, sigmoidDerivative } from "../algorithms/sigmoid";
import createNetwork, { randomNetworkData } from "../network";

const nn = createNetwork(
	randomNetworkData([2, 4, 1]),
	[relu, sigmoid, sigmoid],
	[reluDerivative, sigmoidDerivative, sigmoidDerivative],
);
const xorInputs = [
	[0, 0],
	[0, 1],
	[1, 0],
	[1, 1],
];
const xorTargets = [[0], [1], [1], [0]];
nn.train(
	xorInputs.map((input, i) => ({ inputs: input, expected: xorTargets[i]! })),
	0.1,
	10000,
);
xorInputs.forEach((input, i) => {
	console.log(
		`Input: ${input}, Target: ${xorTargets[i]}, Predicted: ${nn.forward(input)}`,
	);
});
