export function pipe<T>(...fns: ((arg: T) => T)[]): (initialValue: T) => T {
	return (initialValue: T) =>
		fns.reduce((value, fn) => fn(value), initialValue);
}
