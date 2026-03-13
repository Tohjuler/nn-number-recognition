
interface DatasetOptions {
    limit?: number;
    offset?: number;
    digits?: number[];
}

export default async function getDataset(type: "train" | "test", options?: DatasetOptions): Promise<{ inputs: number[]; expected: number[] }[]> {
    const { limit, offset = 0, digits } = options || {};
    const dataset: { inputs: number[]; expected: number[] }[] = [];
    const digitSet = new Set(digits);

    for (let digit = 0; digit <= 9; digit++) {
        if (digits && !digitSet.has(digit)) continue;

        const data = await Bun.file(path(type, digit)).json() as number[][];
        const digitData = data.slice(offset, limit ? offset + limit : undefined).map((inputs) => ({
            inputs,
            expected: Array(10).fill(0).map((_, i) => (i === digit ? 1 : 0)),
        }));
        dataset.push(...digitData);
    }
    return dataset;
}

function path(type: "train" | "test", digit: number): string {
    return `./src/mnist-data/assets/${type}-digits/${digit}.json`;
}