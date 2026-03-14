import { softmax } from "./softmax"; // wherever you put it

export function softmaxCrossEntropyLossWithDelta(
	logits: number[],
	expectedOneHot: number[],
) {
	const probs = softmax(logits);

	// loss = -sum y_i log(p_i)
	const eps = 1e-12;
	let loss = 0;
	for (let i = 0; i < probs.length; i++) {
		const p = Math.max(eps, probs[i]!);
		loss += -expectedOneHot[i]! * Math.log(p);
	}

	// deltaZ = probs - y
	const deltaZ = probs.map((p, i) => p - expectedOneHot[i]!);

	return { loss, deltaZ };
}

export function softmaxCrossEntropyLoss(logits: number[], expectedOneHot: number[]) {
	const probs = softmax(logits);

	// loss = -sum y_i log(p_i)
	const eps = 1e-12;
	let loss = 0;
	for (let i = 0; i < probs.length; i++) {
		const p = Math.max(eps, probs[i]!);
		loss += -expectedOneHot[i]! * Math.log(p);
	}
	return loss;
}
