

async function main() {
    const path = parseArg("path", "./models/training-results-full-shuffle-low-lr-10.json");
    
    const file = Bun.file(path);

    if (!(await file.exists())) {
        console.error(`File not found: ${path}`);
        return;
    }

    const res = await file.json() as {
        epoch: number;
        times: Record<string, number>;
        lossOverEpochs: number[];
        validationLossOverEpochs?: number[];
        validationAccuracyOverEpochs?: number[];
    };

    console.log("\nTime breakdown:");
	console.table(
		Object.entries(res.times).map(([key, value]) => ({
			Operation: key,
			Time: formatTime(value),
		})),
	);

	console.log("\nFinal Metrics:");
	console.table(
		Array.from({ length: res.epoch }, (_, i) => ({
			Epoch: i + 1,
			"Average Loss": res.lossOverEpochs[i]!,
			"Validation Loss": res.validationLossOverEpochs
				? res.validationLossOverEpochs[i]
				: "N/A",
			"Validation Accuracy": res.validationAccuracyOverEpochs
				? `${(res.validationAccuracyOverEpochs[i]! * 100).toFixed(2)}%`
				: "N/A",
		})),
	);
}

main()

function formatTime(ms: number): string {
	const seconds = Math.floor(ms / 1000);
	const minutes = Math.floor(seconds / 60);
	const hours = Math.floor(minutes / 60);

	if (hours > 0) {
		return `${hours}h ${minutes % 60}m ${seconds % 60}s`;
	}
	if (minutes > 0) {
		return `${minutes}m ${seconds % 60}s`;
	}
	if (seconds > 0) {
		return `${seconds}s ${Math.round(ms % 1000)}ms`;
	}

	return `${ms.toFixed(6)}ms`;
}

function parseArg(name: string, defaultValue: string): string {
	const arg = process.argv.find((arg) => arg.startsWith(`--${name}=`));
	if (arg) {
		return arg.split("=")[1]!;
	}
	return defaultValue;
}