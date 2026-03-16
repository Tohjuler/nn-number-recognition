import { softmax } from "./softmax"; // wherever you put it

export function softmaxCrossEntropyLossWithDelta(
	logits: number[],
	expectedOneHot: number[],
) {
	const probs = softmax(logits);

	return crossEntropyLossWithDelta(probs, expectedOneHot);
}

export function softmaxCrossEntropyLoss(logits: number[], expectedOneHot: number[]) {
	const probs = softmax(logits);

	return crossEntropyLoss(probs, expectedOneHot);
}

export function crossEntropyLoss(logits: number[], expectedOneHot: number[]) {
	// loss = -sum y_i log(p_i)
	const eps = 1e-12;
	let loss = 0;
	for (let i = 0; i < logits.length; i++) {
		const p = Math.max(eps, logits[i]!);
		loss += -expectedOneHot[i]! * Math.log(p);
	}
	return loss;
}

export function crossEntropyLossWithDelta(logits: number[], expectedOneHot: number[]) {
	// loss = -sum y_i log(p_i)
	const eps = 1e-12;
	let loss = 0;
	for (let i = 0; i < logits.length; i++) {
		const p = Math.max(eps, logits[i]!);
		loss += -expectedOneHot[i]! * Math.log(p);
	}

	// deltaZ = probs - y
	const deltaZ = logits.map((p, i) => p - expectedOneHot[i]!);

	return { loss, deltaZ };
}
