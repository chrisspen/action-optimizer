# 🤖 AGENTS.md

This file provides guidance and context for AI coding agents working in this repository.
It describes the project structure, conventions, workflows, and constraints to help the agent produce high-quality, consistent code.

---

## 📦 Project Overview

**Name:**
Action Optimizer

**Description:**
This tool processes a daily log spreadsheet of activities and makes recommendations to optimize a global fitness heuristic function.

**Main technologies:**
Python 3.10, Weka, NumPy, SciPy, Scikit-Learn, Pandas

**Primary goals:**
Maintain consistent code style, ensure tests pass.

---

## 🗂️ Directory Structure

| Directory | Purpose |
|------------|----------|
| `/action_optimizer` | Main source code |
| `/action_optimizer/tests` | Unit and integration tests |
| `/action_optimizer/autofill.py` | Utility for bulk autofill operations in a spreadsheet |
| `/action_optimizer/optimizer.py` | Main analysis script |
| `/reports` | Destination output for report generation (not version controlled) |
| `/scripts` | Utility scripts |
| `/init_virtualenv.sh` | Initializes the .env Python virtual environment for running all scripts from |

*(Adjust as needed.)*

---

## 🧩 Code Conventions

**Language and style:**
- Follow Yapf and Pylint configuration as specified in .pre-commit-config.yaml
- Use single quotes (`'`) for strings.
- Use 4 spaces for indentation.
- Write descriptive docstrings or JSDoc comments for public functions.

**Naming patterns:**
- Classes: `PascalCase`
- Functions & variables: `snake_case` (Python) / `camelCase` (JS)
- Constants: `UPPER_SNAKE_CASE`

**File organization:**
- Each module has a clear, single responsibility.
- Avoid circular imports.
- Keep functions small and composable.

---

## 🧪 Testing

**How to run tests:**

You can run all tests across multiple environments using Tox with:

```bash
test.sh
```

Or run a speific test with:

```bash
python -m unittest action_optimizer.tests.test_autofill.Tests.test_autofill
```
