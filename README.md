# Knowledgator Documentation

This repository contains the source code for the Knowledgator documentation site, built with Docusaurus.

Below you'll find instructions for setup, development, and contributing.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Using Yarn](#using-yarn)
  - [Using npm](#using-npm)
- [Local Development](#local-development)
- [Building for Production](#building-for-production)
- [Serving the Production Build](#serving-the-production-build)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **Node.js** (>= 18.0.x)
- **npm** (comes bundled with Node.js) or **yarn** (>= 1.22.x)
- A terminal / command prompt

#### Installing Node.js and npm

Node.js includes npm by default. You can install Node.js from the [official website](https://nodejs.org/en/download).

To verify installation:

```bash
node --version # e.g., v23.8.0
npm --version # e.g., 11.1.0
```

#### Installing yarn (optional)

Yarn is an alternative package manager.

It is recommended to install Yarn through the npm package manager, which comes bundled with Node.js when you install it on your system:

```bash
npm install --global yarn
```

Verify Yarn installation:

```bash
yarn --version # e.g., 1.22.22
```

---

## Installation

Clone this repository and install dependencies.

```bash
git clone git@github.com:Knowledgator/Documentation.git
cd Documentation
```

#### Using Yarn

Install dependencies:

```bash
yarn install
```

#### Using npm

Install dependencies:

```bash
npm install
```

---

## Local Development

Start a local development server and open the site in your default browser (live reload is enabled):

```bash
# With yarn
yarn start

# With npm
npm start
```

This will start the server at [http://localhost:3000](http://localhost:3000) by default.

---

## Building for Production

Generate optimized static files into the `build` directory.

```bash
# With yarn
yarn build

# With npm
npm build
```

The production-ready files will be located in the `build` folder.

---

## Serving the Production Build

You can serve the static build locally to verify the output:

```bash
# Using Yarn
yarn serve

# Using npm
npm serve
```

This serves your built site on [http://localhost:5000](http://localhost:5000) by default.

---

## Contributing

Contributions are welcome. Please open issues and submit pull requests for improvements.

Make sure to run the project locally to verify your changes before submitting them.
