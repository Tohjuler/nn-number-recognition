type Prediction = {
	index: number;
	probability: number;
};

export function renderPrediction(
	probabilityList: HTMLUListElement,
	predictedDigitEl: HTMLElement,
	predictedConfidenceEl: HTMLElement,
	probabilities: number[],
): void {
	const ranked = probabilities
		.map((probability, index) => ({ index, probability }))
		.sort((a, b) => b.probability - a.probability);

	const top = ranked[0];
	if (!top) {
		renderPredictionEmpty(probabilityList, predictedDigitEl, predictedConfidenceEl);
		return;
	}

	predictedDigitEl.textContent = String(top.index);
	predictedConfidenceEl.textContent = `${(top.probability * 100).toFixed(2)}% confidence`;

	probabilityList.innerHTML = "";
	for (const { index, probability } of ranked as Prediction[]) {
		const item = document.createElement("li");
		item.className = "probability-item";
		item.innerHTML = `
			<span class="digit-label">${index}</span>
			<div class="bar"><div class="fill" style="width:${(probability * 100).toFixed(2)}%"></div></div>
			<span class="pct">${(probability * 100).toFixed(2)}%</span>
		`;
		probabilityList.append(item);
	}
}

export function renderPredictionEmpty(
	probabilityList: HTMLUListElement,
	predictedDigitEl: HTMLElement,
	predictedConfidenceEl: HTMLElement,
): void {
	predictedDigitEl.textContent = "-";
	predictedConfidenceEl.textContent = "Draw and predict";
	probabilityList.innerHTML = "";

	for (let digit = 0; digit < 10; digit++) {
		const item = document.createElement("li");
		item.className = "probability-item";
		item.innerHTML = `
			<span class="digit-label">${digit}</span>
			<div class="bar"><div class="fill" style="width:0%"></div></div>
			<span class="pct">0.00%</span>
		`;
		probabilityList.append(item);
	}
}
