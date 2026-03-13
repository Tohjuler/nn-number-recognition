export default function mse(predicted: number[], actual: number[]): number {
    if (predicted.length !== actual.length) throw new Error("Predicted and actual arrays must have the same length.");
	return (
		predicted.reduce((sum, p, i) => sum + (p - actual[i]!) ** 2, 0) /
		predicted.length
	);
}
