export const APP_TEMPLATE = `
	<div class="page-glow"></div>
	<main class="app-shell">
		<header>
			<h1>Digit Tester</h1>
			<p class="subtitle">
				Draw a digit from 0 to 9, run inference, and inspect confidence scores from the trained model.
			</p>
		</header>

		<section class="panel-grid">
			<section class="panel draw-panel">
				<div class="panel-head">
					<h2>Input</h2>
					<p>Canvas is converted to 28x28 grayscale before inference.</p>
				</div>

				<canvas id="draw-canvas" width="280" height="280" aria-label="Digit drawing canvas"></canvas>

				<div class="controls">
					<label class="field">
						<span>Brush size</span>
						<input id="brush-size" type="range" min="6" max="36" step="1" value="16" />
						<strong id="brush-size-value">16</strong>
					</label>

					<label class="checkbox">
						<input id="auto-center" type="checkbox" checked />
						<span>Auto-center before prediction</span>
					</label>

					<div class="button-row">
						<button id="predict-btn" class="btn btn-primary" type="button">Predict</button>
						<button id="center-btn" class="btn" type="button">Center Digit</button>
						<button id="clear-btn" class="btn" type="button">Clear</button>
					</div>
				</div>
			</section>

			<section class="panel result-panel">
				<div class="panel-head">
					<h2>Prediction</h2>
					<p id="model-status">Loading model...</p>
				</div>

				<div class="prediction-box">
					<p class="label">Top prediction</p>
					<p id="predicted-digit" class="predicted-digit">-</p>
					<p id="predicted-confidence" class="predicted-confidence">Draw and predict</p>
				</div>

				<div class="preview-wrap">
					<p class="label">28x28 preview</p>
					<canvas id="preview-canvas" width="28" height="28" aria-label="28 by 28 preview"></canvas>
				</div>

				<div>
					<p class="label">Class probabilities</p>
					<ul id="probability-list" class="probability-list"></ul>
				</div>
			</section>
		</section>
	</main>
`;
