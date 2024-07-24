# Contributing to Scikit-Ollama

Welcome to the contributing guide for our project documentation. This document aims to provide all the necessary information for anyone looking to contribute. Whether you're fixing a typo, improving the docs, or adding a new section, your contributions are greatly appreciated.

You can contribute at two major parts: 1. the [documentation](#contribute-to-documentation), or 2. the [code](#contribute-to-code). We will show how to do both!

## Contribute to documentation

The documentation uses scikit-llm's [documentation repository](https://github.com/BeastByteAI/skllm-docs) as the baseline. We've made design alterations to ensure our work remains distinct. If you're interested in contributing here, consider also contributing to [scikit-llm](https://github.com/iryna-kondr/scikit-llm).

### Environment Setup

1. **Install NVM for Node.js version management**:
   ```bash
   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/master/install.sh | bash
   ```
2. **Install and use the required Node.js version**:
   ```bash
   nvm install 22 --verbose # get the required node version
   nvm use 22 # set it as default
   ```
3. **Install Next**:
   ```bash
   npm install --save next
   ```
4. **Install `serve` to serve the static site locally** (assumes Debian-based Linux):
   ```bash
   npm install -g serve
   ```
5. **Build and serve the static site**:
   ```bash
   npm run build # build the static site locally
   serve -s out # serve the site locally
   ```

### How to Contribute

1. Fork the repository.
2. Make your changes.
3. Test locally.
4. Submit a pull request

### Additional structure information

Under `src/app/docs/` you can find the directories for all sub-pages. Each sub-page includes a `page.md` that is later rendered with markdoc. Simply adjust the mardkown file or create a new folder if you have an additional topic to contribute!

Under `src/lib/navigation.js` you can find the navigation section. New documentation should be included in the links.

## Contribute to code

For the code fork the repository as usual and then follow the below steps.

Install dependencies:

```bash
pip install -r requirements-dev.txt
pre-commit install
```

Once you've made all changes please check that your code conforms to the style guidelines set in the pyproject.toml. You can simply run this command to autoformat and check everything:

```bash
pre-commit run --all-files
```

You may have to run it twice if any file changes were made.
