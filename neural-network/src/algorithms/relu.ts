export function relu(x: number): number {
	return Math.max(0, x);
}

export const reluDerivative = (x: number): number => (x > 0 ? 1 : 0);
