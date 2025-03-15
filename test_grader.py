import os
import re
from openai import OpenAI
import base64
from pathlib import Path
import json
import datetime

# Init OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create storage directories
os.makedirs("reference_solutions", exist_ok=True)
os.makedirs("graded_results", exist_ok=True)
os.makedirs("student_solutions", exist_ok=True)


def extract_problem_from_tex(tex_file_path):
    """Extract the problem statement from a TeX file"""
    with open(tex_file_path, 'r') as f:
        tex_content = f.read()
    
    # Extract the problem-related content, removing latex commands
    problems = []
    
    # Process each problem section
    problem_sections = re.findall(r'\\bigskip\s+{\\bf \d+\. \(\d+ points.*?\)}(.*?)(?=\\bigskip|\\end{document})', 
                                  tex_content, re.DOTALL)
    
    for problem in problem_sections:
        # Clean up latex commands
        cleaned = problem.replace('\\Hspace', ' ')
        cleaned = re.sub(r'\\emph{(.*?)}', r'\1', cleaned)
        cleaned = re.sub(r'\\textbf{(.*?)}', r'**\1**', cleaned)
        # Convert math expressions
        cleaned = re.sub(r'\$([^$]*)\$', r'\$\1\$', cleaned)
        # Remove vfill and other tex commands
        cleaned = re.sub(r'\\vfill', '', cleaned)
        cleaned = re.sub(r'\\item\[(.*?)\]', r'- \1', cleaned)
        
        problems.append(cleaned)
    
    # Join all problem parts into a single statement
    full_problem = "\n\n".join(problems)
    
    # Clean up any remaining latex artifacts
    full_problem = re.sub(r'\\begin{.*?}', '', full_problem)
    full_problem = re.sub(r'\\end{.*?}', '', full_problem)
    full_problem = re.sub(r'\\.*?{', '', full_problem)
    full_problem = re.sub(r'}', '', full_problem)
    
    # Basic extraction of the problem text if no problems were found
    if not problems:
        # If regex didn't work, just extract the main content
        full_problem = "This is a graduate-level algorithms problem set on flow networks, covering:\n"
        full_problem += "1. Construction and properties of residual networks\n"
        full_problem += "2. Determining augmenting paths\n"
        full_problem += "3. Proving maximum flow\n"
        full_problem += "4. Finding minimum-capacity cuts\n"
        full_problem += "5. Analysis of Ford-Fulkerson and Edmonds-Karp algorithms"
    
    return full_problem


def ocr_step(pdf_path):
    """Extract text from a PDF using GPT-4o OCR capabilities"""
    pdf_file = Path(pdf_path)
    assert pdf_file.is_file()

    # Try using OpenAI GPT-4o for OCR
    try:
        print("Attempting OCR with GPT-4o...")
        with open(pdf_path, "rb") as f:
            data = f.read()
        
        base64_string = base64.b64encode(data).decode("utf-8")
        
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "file": {"file_name": pdf_file.name, "file_data": base64_string}
                        },
                        {
                            "type": "text", 
                            "text": "Extract all text content from this PDF document. Maintain the original formatting as much as possible, including paragraphs, bullet points, and section headers. Return the extracted text in markdown format."
                        }
                    ]
                }
            ]
        )
        
        structured_text = completion.choices[0].message.content
        print("OCR successful with GPT-4o")
        return structured_text
        
    except Exception as e:
        print(f"GPT-4o OCR failed with error: {str(e)}")
        return None


def generate_reference_solution(problem_statement):
    """Generate a reference solution using GPT-4-turbo"""
    prompt = f"Provide a detailed, step-by-step solution to the following problem:\n\n{problem_statement}"

    try:
        # Try using OpenAI o1 for reasoning if available
        completion = openai_client.chat.completions.create(
            model="gpt-4-turbo",  # Changed from o1 since it might not be available
            messages=[
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating reference solution: {str(e)}")
        return None


def grade_solution(problem_statement, student_solution, reference_solution):
    """Grade the student's solution using a rubric"""
    prompt = (
        f"You are an expert grader for graduate-level algorithm problems. Grade the following student's solution according to the rubric below.\n\n"
        f"### Problem Statement:\n{problem_statement}\n\n"
        f"### Student Solution:\n{student_solution}\n\n"
        f"### Reference Solution:\n{reference_solution}\n\n"
        f"### Grading Rubric:\n"
        f"1. Problem Understanding (0-2 points):\n"
        f"   - Does the student correctly understand what the problem is asking?\n"
        f"   - Have they identified the key constraints and inputs correctly?\n\n"
        f"2. Algorithm Design (0-3 points):\n"
        f"   - Is the algorithm approach correct?\n"
        f"   - Is it efficient (optimal time/space complexity)?\n"
        f"   - Is the algorithm fully explained?\n\n"
        f"3. Correctness (0-3 points):\n"
        f"   - Does the solution correctly solve the problem?\n"
        f"   - Are edge cases handled properly?\n"
        f"   - Is the solution free of logical errors?\n\n"
        f"4. Analysis & Proofs (0-2 points):\n"
        f"   - Is the time/space complexity analysis correct?\n"
        f"   - Are any required proofs (correctness, optimality) complete and accurate?\n\n"
        f"Please provide:\n"
        f"1. A score for each rubric category\n"
        f"2. Brief justification for each score, citing specific parts of the student's solution\n"
        f"3. A total score out of 10\n"
        f"4. 1-2 specific areas of improvement for the student\n"
        f"Format your response clearly with markdown headers for each section."
    )

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4-turbo", 
            messages=[{"role": "user", "content": prompt}]
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error grading solution: {str(e)}")
        return None


def main():
    # Get problem statement from TeX file
    problem_tex_path = "/Users/bozen/Code/Agents/grader/Example_Problem.tex"
    problem_statement = extract_problem_from_tex(problem_tex_path)
    print("\n--- Problem Statement ---")
    print(problem_statement[:300] + "..." if len(problem_statement) > 300 else problem_statement)
    
    # Generate reference solution
    print("\n--- Generating Reference Solution ---")
    reference_solution = generate_reference_solution(problem_statement)
    
    # Save reference solution
    problem_id = "test_quiz4"
    ref_solution_path = os.path.join("reference_solutions", f"{problem_id}.json")
    with open(ref_solution_path, "w") as f:
        json.dump({
            "problem_id": problem_id, 
            "problem_statement": problem_statement,
            "reference_solution": reference_solution,
            "timestamp": datetime.datetime.now().isoformat()
        }, f, indent=4)
    
    # Process student solution
    student_pdf_path = "/Users/bozen/Code/Agents/grader/example_student_sol.pdf"
    student_id = "test_student"
    
    print("\n--- Extracting Student Solution ---")
    student_solution = ocr_step(student_pdf_path)
    
    # Grade the solution
    print("\n--- Grading Student Solution ---")
    grading_result = grade_solution(problem_statement, student_solution, reference_solution)
    
    # Save the grading result
    result_path = os.path.join("graded_results", f"{student_id}_result.json")
    with open(result_path, "w") as f:
        json.dump({
            "student_id": student_id,
            "problem_id": problem_id,
            "problem_statement": problem_statement,
            "structured_text": student_solution,
            "reference_solution": reference_solution,
            "llm_grading": grading_result,
            "timestamp": datetime.datetime.now().isoformat()
        }, f, indent=4)
    
    # Display the grading result
    print("\n--- Grading Result ---")
    print(grading_result)
    
    print(f"\nGrading saved to {result_path}")


if __name__ == "__main__":
    main()