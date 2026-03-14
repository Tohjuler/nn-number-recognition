export function pipe<T>(...fns: ((arg: T) => T)[]): (initialValue: T) => T {
	return (initialValue: T) =>
		fns.reduce((value, fn) => fn(value), initialValue);
}

export function timerHelper() {
	const times: Record<string, number> = {};
	const savedTimes: Record<string, number> = {};
	const avgTimes: Record<string, number[]> = {};

	const start = (label: string) => {
		times[label] = performance.now();
	};

	const end = (label: string, avgLabel?: string) => {
		if (times[label] === undefined)
			throw new Error(`No start time recorded for label: ${label}`);

		const duration = performance.now() - times[label]!;
		delete times[label];

		if (avgLabel) {
			if (!avgTimes[avgLabel]) avgTimes[avgLabel] = [];
			avgTimes[avgLabel]!.push(duration);
		}

		return duration;
	};

	const endAndSave = (label: string) => {
		const duration = end(label);
		savedTimes[label] = duration;
		return duration;
	};

	const getOutput = () => {
		const output: Record<string, number> = savedTimes;
		for (const label in avgTimes) {
			const durations = avgTimes[label]!;
			output[label] = durations.reduce((a, b) => a + b, 0) / durations.length;
		}
		return output;
	};

	const getAvg = (label: string) => {
		const durations = avgTimes[label];
		if (!durations || durations.length === 0)
			throw new Error(`No average times recorded for label: ${label}`);
		return durations.reduce((a, b) => a + b, 0) / durations.length;
	};

	return { start, end, endAndSave, getOutput, getAvg };
}

export function shuffle<T>(array: T[]): T[] {
    const arr = array.slice();
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j]!, arr[i]!];
    }
    return arr;
}