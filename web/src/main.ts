import "./style.css";
import "@fontsource/maple-mono/index.css";

import { CanvasController } from "./canvas-controller";
import { MODELS, MODEL_VECTOR_LENGTH } from "./constants";
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
const modelSelect = getElement<HTMLSelectElement>("model-select");
const modelUpload = getElement<HTMLInputElement>("model-upload");

const modelEntries = Object.entries(MODELS);
const defaultModelLabel = modelEntries[0]?.[0] ?? null;

const canvasController = new CanvasController(
	drawCanvas,
	get2d(drawCanvas),
	get2d(previewCanvas),
);
const modelService = new NeuralModelService();

canvasController.bindDrawing(() => {
	canvasController.renderPreview(canvasController.captureInputVector());
});

initializeModelSelect();
canvasController.renderPreview(Array(MODEL_VECTOR_LENGTH).fill(0));
renderPredictionEmpty(probabilityList, predictedDigitEl, predictedConfidenceEl);
void loadInitialModel();

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

modelSelect.addEventListener("change", () => {
	const selectedLabel = modelSelect.value;
	if (!selectedLabel || !(selectedLabel in MODELS)) {
		return;
	}

	void loadBuiltInModel(selectedLabel);
});

modelUpload.addEventListener("change", () => {
	const selectedFile = modelUpload.files?.[0];
	if (!selectedFile) {
		return;
	}

	void loadUploadedModel(selectedFile);
});

function initializeModelSelect(): void {
	for (const [label] of modelEntries) {
		const option = document.createElement("option");
		option.value = label;
		option.textContent = label;
		modelSelect.append(option);
	}

	if (defaultModelLabel) {
		modelSelect.value = defaultModelLabel;
	}
}

async function loadInitialModel(): Promise<void> {
	if (!defaultModelLabel) {
		modelStatus.textContent = "No built-in models configured.";
		return;
	}

	await loadBuiltInModel(defaultModelLabel);
}

async function loadBuiltInModel(modelLabel: string): Promise<void> {
	const modelPath = MODELS[modelLabel as keyof typeof MODELS];
	if (!modelPath) {
		modelStatus.textContent = "Unknown model selected.";
		return;
	}

	modelStatus.textContent = `Loading ${modelLabel}...`;

	try {
		const layers = await modelService.load(modelPath);
		modelStatus.textContent = `${modelLabel} ready (${layers.length} layers).`;
	} catch (error) {
		modelStatus.textContent =
			error instanceof Error ? error.message : "Failed to load model.";
	}
}

async function loadUploadedModel(file: File): Promise<void> {
	modelStatus.textContent = `Loading custom model (${file.name})...`;

	try {
		const text = await file.text();
		const parsed: unknown = JSON.parse(text);
		const layers = await modelService.loadUploaded(parsed, file.name);
		modelStatus.textContent = `Custom model ready (${layers.length} layers).`;
	} catch (error) {
		modelStatus.textContent =
			error instanceof Error ? error.message : "Failed to load custom model.";
	} finally {
		modelUpload.value = "";
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
