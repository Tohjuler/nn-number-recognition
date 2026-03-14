import {
	CANVAS_SIZE,
	DRAW_BACKGROUND,
	DRAW_FOREGROUND,
	MODEL_INPUT_SIZE,
	MODEL_VECTOR_LENGTH,
} from "./constants";

export class CanvasController {
	private drawing = false;
	private brushSize = 16;
	private lastPoint: { x: number; y: number } | null = null;
	private readonly drawCanvas: HTMLCanvasElement;
	private readonly drawCtx: CanvasRenderingContext2D;
	private readonly previewCtx: CanvasRenderingContext2D;

	constructor(
		drawCanvas: HTMLCanvasElement,
		drawCtx: CanvasRenderingContext2D,
		previewCtx: CanvasRenderingContext2D,
	) {
		this.drawCanvas = drawCanvas;
		this.drawCtx = drawCtx;
		this.previewCtx = previewCtx;
		this.initializeCanvas();
	}

	setBrushSize(value: number): void {
		this.brushSize = value;
	}

	bindDrawing(onStrokeEnd: () => void): void {
		this.drawCanvas.addEventListener("pointerdown", (event) => {
			this.drawing = true;
			this.drawCanvas.setPointerCapture(event.pointerId);
			const point = this.pointerToCanvasPoint(event);
			this.lastPoint = point;
			this.drawDot(point.x, point.y);
		});

		this.drawCanvas.addEventListener("pointermove", (event) => {
			if (!this.drawing || !this.lastPoint) return;
			const point = this.pointerToCanvasPoint(event);
			this.drawStroke(this.lastPoint, point);
			this.lastPoint = point;
		});

		this.drawCanvas.addEventListener("pointerup", () => {
			this.drawing = false;
			this.lastPoint = null;
			onStrokeEnd();
		});

		this.drawCanvas.addEventListener("pointerleave", () => {
			this.drawing = false;
			this.lastPoint = null;
		});

		this.drawCanvas.addEventListener("touchstart", (event) => {
			event.preventDefault();
		});
	}

	clearCanvas(): void {
		this.drawCtx.fillStyle = DRAW_BACKGROUND;
		this.drawCtx.fillRect(0, 0, this.drawCanvas.width, this.drawCanvas.height);
	}

	captureInputVector(): number[] {
		const imageData = this.drawCtx.getImageData(
			0,
			0,
			this.drawCanvas.width,
			this.drawCanvas.height,
		).data;
		const bin = CANVAS_SIZE / MODEL_INPUT_SIZE;
		const output = new Array<number>(MODEL_VECTOR_LENGTH).fill(0);

		for (let y = 0; y < MODEL_INPUT_SIZE; y++) {
			for (let x = 0; x < MODEL_INPUT_SIZE; x++) {
				let sum = 0;
				for (let sy = 0; sy < bin; sy++) {
					for (let sx = 0; sx < bin; sx++) {
						const srcX = x * bin + sx;
						const srcY = y * bin + sy;
						const idx = (srcY * CANVAS_SIZE + srcX) * 4;
						const r = imageData[idx] ?? 0;
						const g = imageData[idx + 1] ?? 0;
						const b = imageData[idx + 2] ?? 0;
						sum += (r + g + b) / (3 * 255);
					}
				}
				output[y * MODEL_INPUT_SIZE + x] = sum / (bin * bin);
			}
		}

		return output;
	}

	renderPreview(input: number[]): void {
		const image = this.previewCtx.createImageData(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
		for (let i = 0; i < input.length; i++) {
			const value = clamp01(input[i] ?? 0);
			const color = Math.round(value * 255);
			const idx = i * 4;
			image.data[idx] = color;
			image.data[idx + 1] = color;
			image.data[idx + 2] = color;
			image.data[idx + 3] = 255;
		}
		this.previewCtx.putImageData(image, 0, 0);
	}

	renderInputOnCanvas(input: number[]): void {
		const offscreen = document.createElement("canvas");
		offscreen.width = MODEL_INPUT_SIZE;
		offscreen.height = MODEL_INPUT_SIZE;
		const offCtx = offscreen.getContext("2d");
		if (!offCtx) {
			throw new Error("2D context is unavailable.");
		}
		const image = offCtx.createImageData(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);

		for (let i = 0; i < input.length; i++) {
			const value = clamp01(input[i] ?? 0);
			const color = Math.round(value * 255);
			const idx = i * 4;
			image.data[idx] = color;
			image.data[idx + 1] = color;
			image.data[idx + 2] = color;
			image.data[idx + 3] = 255;
		}

		offCtx.putImageData(image, 0, 0);
		this.drawCtx.fillStyle = DRAW_BACKGROUND;
		this.drawCtx.fillRect(0, 0, this.drawCanvas.width, this.drawCanvas.height);
		this.drawCtx.imageSmoothingEnabled = false;
		this.drawCtx.drawImage(offscreen, 0, 0, this.drawCanvas.width, this.drawCanvas.height);
		this.drawCtx.imageSmoothingEnabled = true;
	}

	centerDigitOnCanvas(): boolean {
		const input = this.captureInputVector();
		let mass = 0;
		let sumX = 0;
		let sumY = 0;

		for (let y = 0; y < MODEL_INPUT_SIZE; y++) {
			for (let x = 0; x < MODEL_INPUT_SIZE; x++) {
				const value = input[y * MODEL_INPUT_SIZE + x] ?? 0;
				if (value < 0.05) continue;
				mass += value;
				sumX += x * value;
				sumY += y * value;
			}
		}

		if (mass === 0) return false;

		const centerX = sumX / mass;
		const centerY = sumY / mass;
		const shiftX = (MODEL_INPUT_SIZE - 1) / 2 - centerX;
		const shiftY = (MODEL_INPUT_SIZE - 1) / 2 - centerY;

		const shifted = new Array<number>(MODEL_VECTOR_LENGTH).fill(0);
		for (let y = 0; y < MODEL_INPUT_SIZE; y++) {
			for (let x = 0; x < MODEL_INPUT_SIZE; x++) {
				const sourceX = Math.round(x - shiftX);
				const sourceY = Math.round(y - shiftY);
				if (
					sourceX < 0 ||
					sourceX >= MODEL_INPUT_SIZE ||
					sourceY < 0 ||
					sourceY >= MODEL_INPUT_SIZE
				) {
					continue;
				}

				shifted[y * MODEL_INPUT_SIZE + x] =
					input[sourceY * MODEL_INPUT_SIZE + sourceX] ?? 0;
			}
		}

		this.renderInputOnCanvas(shifted);
		return true;
	}

	private initializeCanvas(): void {
		this.drawCtx.fillStyle = DRAW_BACKGROUND;
		this.drawCtx.fillRect(0, 0, this.drawCanvas.width, this.drawCanvas.height);
		this.drawCtx.lineCap = "round";
		this.drawCtx.lineJoin = "round";
		this.drawCtx.strokeStyle = DRAW_FOREGROUND;
		this.drawCtx.imageSmoothingEnabled = true;
	}

	private drawDot(x: number, y: number): void {
		this.drawCtx.beginPath();
		this.drawCtx.arc(x, y, this.brushSize / 2, 0, Math.PI * 2);
		this.drawCtx.fillStyle = DRAW_FOREGROUND;
		this.drawCtx.fill();
	}

	private drawStroke(
		from: { x: number; y: number },
		to: { x: number; y: number },
	): void {
		this.drawCtx.beginPath();
		this.drawCtx.moveTo(from.x, from.y);
		this.drawCtx.lineTo(to.x, to.y);
		this.drawCtx.lineWidth = this.brushSize;
		this.drawCtx.stroke();
	}

	private pointerToCanvasPoint(event: PointerEvent): { x: number; y: number } {
		const rect = this.drawCanvas.getBoundingClientRect();
		return {
			x: ((event.clientX - rect.left) / rect.width) * this.drawCanvas.width,
			y: ((event.clientY - rect.top) / rect.height) * this.drawCanvas.height,
		};
	}
}

function clamp01(value: number): number {
	if (value < 0) return 0;
	if (value > 1) return 1;
	return value;
}
