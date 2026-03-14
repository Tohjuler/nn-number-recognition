import "./style.css";
import "@fontsource/maple-mono/index.css";

import { CanvasController } from "./canvas-controller";
import { MODEL_URL, MODEL_VECTOR_LENGTH } from "./constants";
import { get2d, getElement } from "./dom";
import { NeuralModelService } from "./model-service";
import { renderPrediction, renderPredictionEmpty } from "./prediction-ui";
import { APP_TEMPLATE } from "./template";

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) throw new Error("Missing #app container.");

app.innerHTML = APP_TEMPLATE;

const drawCanvas = getElement<HTMLCanvasElement>("draw-canvas");
const previewCanvas = getElement<HTMLCanvasElement>("preview-canvas");
const modelStatus = getElement<HTMLParagraphElement>("model-status");
const predictedDigitEl = getElement<HTMLParagraphElement>("predicted-digit");
const predictedConfidenceEl = getElement<HTMLParagraphElement>(
	"predicted-confidence",
);
const probabilityList = getElement<HTMLUListElement>("probability-list");
const brushSizeInput = getElement<HTMLInputElement>("brush-size");
const brushSizeValue = getElement<HTMLElement>("brush-size-value");
const autoCenterInput = getElement<HTMLInputElement>("auto-center");
const predictBtn = getElement<HTMLButtonElement>("predict-btn");
const centerBtn = getElement<HTMLButtonElement>("center-btn");
const clearBtn = getElement<HTMLButtonElement>("clear-btn");

const canvasController = new CanvasController(
	drawCanvas,
	get2d(drawCanvas),
	get2d(previewCanvas),
);
const modelService = new NeuralModelService();

canvasController.bindDrawing(() => {
	canvasController.renderPreview(canvasController.captureInputVector());
});

canvasController.renderPreview(Array(MODEL_VECTOR_LENGTH).fill(0));
renderPredictionEmpty(probabilityList, predictedDigitEl, predictedConfidenceEl);
void loadModel();

brushSizeInput.addEventListener("input", () => {
	const brushSize = Number(brushSizeInput.value);
	canvasController.setBrushSize(brushSize);
	brushSizeValue.textContent = String(brushSize);
});

predictBtn.addEventListener("click", () => {
	void runPrediction();
});

centerBtn.addEventListener("click", () => {
	const centered = canvasController.centerDigitOnCanvas();
	canvasController.renderPreview(canvasController.captureInputVector());
	if (!centered) {
		predictedConfidenceEl.textContent = "No visible digit to center.";
	}
});

clearBtn.addEventListener("click", () => {
	canvasController.clearCanvas();
	canvasController.renderPreview(canvasController.captureInputVector());
	renderPredictionEmpty(
		probabilityList,
		predictedDigitEl,
		predictedConfidenceEl,
	);
});

async function loadModel(): Promise<void> {
	try {
		const layers = await modelService.load(MODEL_URL);
		modelStatus.textContent = `Model ready (${layers.length} layers).`;
	} catch (error) {
		modelStatus.textContent =
			error instanceof Error ? error.message : "Failed to load model.";
	}
}

async function runPrediction(): Promise<void> {
	try {
		if (autoCenterInput.checked) {
			canvasController.centerDigitOnCanvas();
		}

		const input = canvasController.captureInputVector();
		canvasController.renderPreview(input);
		const probabilities = modelService.predict(input);
		renderPrediction(
			probabilityList,
			predictedDigitEl,
			predictedConfidenceEl,
			probabilities,
		);
	} catch (error) {
		predictedConfidenceEl.textContent =
			error instanceof Error ? error.message : "Prediction failed.";
	}
}
