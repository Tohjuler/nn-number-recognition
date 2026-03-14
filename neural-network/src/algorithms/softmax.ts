export function softmax(logits: number[]): number[] {
	// subtract max(logits) for numerical stability
	let max = -Infinity;
	for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i]!);

	const exps = new Array<number>(logits.length);
	let sum = 0;
	for (let i = 0; i < logits.length; i++) {
		const e = Math.exp(logits[i]! - max);
		exps[i] = e;
		sum += e;
	}

	for (let i = 0; i < exps.length; i++) exps[i]! /= sum;
	return exps;
}
