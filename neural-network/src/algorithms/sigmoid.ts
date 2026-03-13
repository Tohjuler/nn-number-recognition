export function sigmoid(x: number): number {
	return 1 / (1 + Math.exp(-x));
}

export function sigmoidDerivative(x: number): number {
	const sx = sigmoid(x);
	return sx * (1 - sx);
}
