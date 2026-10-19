_This project has been created as part of the 42 curriculum by marberge._

<div align="center">
<br>
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTQPzuYKu7n0cWUYa5Kbg0_LrlEQAIURWeo9A&s" alt="42 Logo" width="400" />

  <br>
</div>

# [PROJECT TITLE]

<div align="center">

<img src="https://img.shields.io/badge/Project-000000?style=for-the-badge&logo=42&logoColor=white" alt="42" />
	<img src="https://img.shields.io/badge/Language-Python-00599C?style=for-the-badge&logo=&logoColor=white" alt="C" />
	<img src="https://img.shields.io/badge/Score-0/100-yellow?style=for-the-badge&logoColor=white" alt="C" />
	<br>
	<img src="https://img.shields.io/badge/TAG-darkviolet?style=for-the-badge" alt="GAME" />
<img src="https://img.shields.io/badge/TAG-darkviolet?style=for-the-badge" alt="Algorithms" />
	<img src="https://img.shields.io/badge/TAG-darkviolet?style=for-the-badge" alt="Pathfinding" />
	<img src="https://img.shields.io/badge/TAG-darkviolet?style=for-the-badge" alt="GUI" />
	<img src="https://img.shields.io/badge/TAG-darkviolet?style=for-the-badge" alt="Parsing" />

</div>

## I. Description

To define.

## II. Instructions

### Prerequisites
Before using this template, ensure you have the following installed on your system:
- **Python 3.10+**
- **uv 0.10.12+**

### Quick Start
To set up the environment and run the project for the first time, simply use the following command in your terminal:

	make run

### Makefile Commands Reference
This project is fully automated using Make. Here is the complete list of available commands to manage the project lifecycle:

**Installation & Setup**
- ```make install``` (or **make all**): Initializes the virtual environment (.venv) and synchronizes all dependencies using uv.
- ```make setup```: Checks your Python version and presence of the uv package manager. Exit if both check are not valid.

**Execution & Debugging**
- ```make run```: Executes the main entry point (src/main.py) inside the isolated virtual environment.
- ```make debug```: Launches the project using the Python Debugger (pdb), allowing you to step through your code line by line.

**Quality & Testing**
- ```make lint```: Runs flake8 for style checking and mypy for static type checking to ensure code quality.
- ```make lint-strict```: Runs the linters but enforces strict typing rules with mypy.
- ```make test```: Runs the entire test suite using pytest.
- ```make test-file ARGS=path/to/test.py```: Runs a specific test file. Replace the FILE variable with your target.

**Building & Cleaning**
- ```make build```: Packages the project into distributable files inside a dist/ directory.
- ```make clean```: Removes all temporary files, such as __pycache__ folders and linter caches.
- ```make fclean```: Performs a deep clean. It executes the clean rule and also removes the virtual environment and build files.
- ```make re```: Rebuilds the project from scratch by running fclean followed by all.

***

## III. Resources

To define.

## IV. Additional sections

To define.
