# LLM Automatic Grading System

An intelligent grading system that uses LLMs to automate the grading of student solutions to graduate-level algorithm problems.

## Features

- Generate reference solutions for problems using LLMs (GPT-4-turbo or GPT-4o)
- Extract text from student PDF submissions using Mistral OCR
- Grade student solutions against reference solutions with a detailed rubric
- Human-in-the-loop verification for each grading
- Storage of reference solutions and graded results for later reference
- Persistent workflow that retains results between sessions

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up API keys in your environment:
   ```
   export OPENAI_API_KEY=your_openai_key
   export MISTRAL_API_KEY=your_mistral_key
   ```

## Usage

Run the main program:

```bash
python main.py --problem_id hw1_problem2
```

To use the advanced reasoning capabilities (OpenAI o1) for reference solution generation:

```bash
python main.py --problem_id hw1_problem2
```

To input a LaTeX problem statement file:

```bash
python main.py --problem_id hw1_problem2 --tex_file path/to/problem.tex
```

To choose OCR provider (GPT-4o or Mistral):

```bash
python main.py --problem_id hw1_problem2 --ocr gpt4o  # Default option
python main.py --problem_id hw1_problem2 --ocr mistral
```

Example with the sample files using Mistral OCR:

```bash
python main.py --problem_id quiz4 --tex_file Example_Problem.tex --ocr mistral
```

### Workflow

1. **First Run with New Problem**:
   - Enter a problem ID (e.g., "hw1_problem2")
   - Provide the problem statement
   - The system generates and stores a reference solution

2. **Processing Student Submissions**:
   - Enter the student ID
   - Provide the path to the student's PDF solution
   - The system extracts text using OCR
   - The LLM grades the solution
   - You review and approve or modify the grading
   - Results are stored for future reference

3. **Subsequent Runs**:
   - The system loads the existing problem statement and reference solution
   - You can immediately start processing student submissions

## Project Structure

- `reference_solutions/`: Stored reference solutions for problems
- `graded_results/`: Results from grading student solutions
- `student_solutions/`: Copies of student PDFs that have been processed

## Customization

You can modify the grading rubric in the `llm_grading_step` function to better suit your specific course needs.

## Requirements

- Python 3.9 or higher
- OpenAI API access
- Mistral API access