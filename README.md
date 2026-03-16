
# NN Number Recognition

This is a implementation of a neural network in TS, from scratch. To test it out I created a simple web UI that for number recognition and trained the nn for that using the MNIST dataset.
As this project was made to learn about neural networks, the web ui was created usign AI. Ofcause it was lookover by me and the style was fully rewritten.

## Setup

The project is made with [bun](https://bun.sh) as a monorepo, the neural network implementaion is in `neural-network/`, the trainer for MNIST is in `trainer/` and the web ui is in `web/`

## Run

Install all the packages:

```bash
bun install
```

### Web UI

Go into `web/`

```bash
bun run build
```

Then the build is in `dist/`

To run the dev version use:

```bash
bun run dev
```

It should then run on <http://localhost:5173>
