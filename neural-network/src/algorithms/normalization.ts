export function normalize(data: number[], min: number, max: number): number[] {
	return data.map((value) => (value - min) / (max - min));
    // Example: If min = 0 and max = 255, then a value of 128 would be normalized to (128 - 0) / (255 - 0) = 128 / 255 ≈ 0.502.
}

export function denormalize(
	normalizedData: number[],
	min: number,
	max: number,
): number[] {
	return normalizedData.map((value) => value * (max - min) + min);
}
