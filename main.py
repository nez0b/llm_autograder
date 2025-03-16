from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from typing import TypedDict, List, Dict, Optional
import json
import os
import re
from mistralai import Mistral, DocumentURLChunk
from mistralai.models import OCRResponse
from pathlib import Path
import base64
from openai import OpenAI
import datetime
import argparse
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# State definition for LangGraph
class GradingState(TypedDict):
    pdf_path: str
    student_id: str
    structured_text: str
    problem_statement: str
    reference_solution: str
    llm_grading: str
    final_grading: Optional[str]
    timestamp: str

# Initialize API clients with environment variables
mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4-turbo")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create storage directories
os.makedirs("reference_solutions", exist_ok=True)
os.makedirs("graded_results", exist_ok=True)
os.makedirs("student_solutions", exist_ok=True)

# Step 1A: Generate reference solution using traditional LLM (GPT-4)
def generate_reference_solution_llm(state: GradingState):
    problem_statement = state['problem_statement']

    prompt = f"Provide a detailed, step-by-step solution to the following problem:\n\n{problem_statement}"

    response = llm.invoke(prompt)
    state['reference_solution'] = response.content

    return state

# Step 1B: Generate reference solution using multimodal LLM (GPT-4o)
def generate_reference_solution_multimodal(state: GradingState):
    pdf_path = state['pdf_path']
    problem_statement = state['problem_statement']

    with open(pdf_path, "rb") as f:
        data = f.read()

    base64_string = base64.b64encode(data).decode("utf-8")

    completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "file", "file": {"file_name": pdf_path, "file_data": base64_string}},
                    {"type": "text", "text": f"Provide a detailed solution to the following problem:\n\n{problem_statement}"}
                ]
            }
        ]
    )

    state['reference_solution'] = completion.choices[0].message.content

    return state

# Generate reference solution using OpenAI reasoning model (o1)
def generate_reference_solution_reasoning(state: GradingState):
    problem_statement = state['problem_statement']

    prompt = f"Provide a detailed, step-by-step solution to the following problem (Work on each problem separately):\n\n{problem_statement}"

    completion = openai_client.chat.completions.create(
        model="o3-mini",
        reasoning_effort="high",
        messages=[
            {
                "role": "user", 
                "content": prompt
            }
        ]
    )

    state['reference_solution'] = completion.choices[0].message.content
    
    # Log the reasoning tokens used (optional)
    reasoning_tokens = completion.usage.completion_tokens_details.reasoning_tokens
    print(f"Reasoning tokens used: {reasoning_tokens}")
    
    return state

# Helper function for OCR step
def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    for img_name, base64_str in images_dict.items():
        markdown_str = markdown_str.replace(f"![{img_name}]({img_name})", f"![{img_name}]({base64_str})")
    return markdown_str

def get_combined_markdown(ocr_response: OCRResponse) -> str:
  markdowns: list[str] = []
  for page in ocr_response.pages:
    image_data = {}
    for img in page.images:
      image_data[img.id] = img.image_base64
    markdowns.append(replace_images_in_markdown(page.markdown, image_data))

  return "\n\n".join(markdowns)

# Step 2: OCR extraction function (with choice of provider)
def ocr_step(state: GradingState):
    pdf_path = state['pdf_path']
    pdf_file = Path(pdf_path)
    assert pdf_file.is_file()

    # Get OCR provider preference from state or use default
    ocr_provider = state.get('ocr_provider', 'gpt4o')  # Default to GPT-4o if not specified
    
    structured_text = None
    
    # Use GPT-4o for OCR
    if ocr_provider == 'gpt4o':
        try:
            print("Performing OCR with GPT-4o...")
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
            
        except Exception as e:
            print(f"GPT-4o OCR failed with error: {str(e)}. Falling back to Mistral OCR...")
            ocr_provider = 'mistral'  # Fall back to Mistral
    
    # Use Mistral OCR
    if ocr_provider == 'mistral' or structured_text is None:
        try:
            print("Performing OCR with Mistral OCR...")
            uploaded_file = mistral_client.files.upload(
                file={"file_name": pdf_file.stem, "content": pdf_file.read_bytes()},
                purpose="ocr",
            )

            signed_url = mistral_client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)

            ocr_response = mistral_client.ocr.process(
                document=DocumentURLChunk(document_url=signed_url.url),
                model="mistral-ocr-latest",
                include_image_base64=True
            )

            #structured_text = "\n\n".join(page.markdown for page in ocr_response.pages)
            structured_text = get_combined_markdown(ocr_response)
            print(structured_text)
            print("OCR completed with Mistral OCR")
        except Exception as e:
            print(f"Mistral OCR failed with error: {str(e)}")
            if structured_text is None:
                raise Exception("All OCR methods failed. Cannot proceed with grading.")
    
    state['structured_text'] = structured_text
    return state

# Step 3: LLM grading function with improved rubric using reasoning model
def llm_grading_step(state: GradingState):
    structured_text = state['structured_text']
    reference_solution = state['reference_solution']
    problem_statement = state['problem_statement']

    prompt = (
        f"You are an expert grader for graduate-level algorithm problems. Grade the following student's solution according to the rubric below.\n\n"
        f"### Problem Statement:\n{problem_statement}\n\n"
        f"### Student Solution:\n{structured_text}\n\n"
        f"### Reference Solution:\n{reference_solution}\n\n"
        f"### Grading Rubric:\n"
        f"1. Problem Attempt:\n"
        f"   - If no attempt then give zero to that sub problem?\n"
        f"2. Correctness (0-10 points) for each problem :\n"
        f"   - Does the solution correctly solve the problem?\n"
        f"   - Is the solution free of logical errors?\n\n"
        f"Please provide:\n"
        f"1. A score for each rubric category (for each problem)\n"
        f"2. Justification for each score, citing specific parts of the student's solution. If points are deducted, give detailed comments.\n"
        f"3. A total score out of 10\n"
        f"Format your response clearly with markdown headers for each section."
    )

    # Use o1 reasoning model instead of standard LLM
    completion = openai_client.chat.completions.create(
        model="o3-mini",
        reasoning_effort="high",
        messages=[
            {
                "role": "user", 
                "content": prompt
            }
        ]
    )

    state['llm_grading'] = completion.choices[0].message.content
    
    # Log the reasoning tokens used
    reasoning_tokens = completion.usage.completion_tokens_details.reasoning_tokens
    print(f"Grading reasoning tokens used: {reasoning_tokens}")
    
    return state

# Step 4: Human-in-the-loop verification
def human_verification_step(state: GradingState):
    print("\n" + "-"*80)
    print("STUDENT ID:", state['student_id'])
    print("-"*80)
    print("\nSTUDENT SOLUTION SUMMARY:")
    print("-"*40)
    print(state['structured_text'][:300] + "..." if len(state['structured_text']) > 300 else state['structured_text'])
    print("-"*40)
    print("\nLLM'S SUGGESTED GRADING:")
    print("-"*40)
    print(state['llm_grading'])
    print("-"*40)

    human_judgment = input("\nEnter your verification (accept/edit the grading) or 'q' to quit: ")
    
    if human_judgment.lower() == 'q':
        print("Exiting grading process.")
        exit(0)
        
    state['final_grading'] = human_judgment
    state['timestamp'] = datetime.datetime.now().isoformat()
    
    # Save graded solution
    result_path = os.path.join("graded_results", f"{state['student_id']}_{int(time.time())}.json")
    with open(result_path, "w") as f:
        json.dump({k: v for k, v in state.items() if k != 'pdf_path'}, f, indent=4)
    
    print(f"Grading saved to {result_path}")
    
    return state

# Build LangGraph pipeline 
graph = StateGraph(GradingState)
graph.add_node("generate_reference_solution_reasoning", generate_reference_solution_reasoning)
graph.add_node("ocr_step", ocr_step)
graph.add_node("llm_grading_step", llm_grading_step)
graph.add_node("human_verification", human_verification_step)

# Set entry point to the reasoning model
graph.set_entry_point("generate_reference_solution_reasoning")  

graph.add_edge("generate_reference_solution_reasoning", "ocr_step")
graph.add_edge("ocr_step", "llm_grading_step")
graph.add_edge("llm_grading_step", "human_verification")
graph.set_finish_point("human_verification")

app = graph.compile()

# Function to extract problem statement from TeX file
def extract_problem_from_tex(tex_file_path):
    """Extract the problem statement from a LaTeX file"""
    with open(tex_file_path, 'r') as f:
        tex_content = f.read()
    
    # Extract the content between document tags
    document_match = re.search(r'\\begin{document}(.*?)\\end{document}', tex_content, re.DOTALL)
    if not document_match:
        return "Problem content extraction failed."
    
    content = document_match.group(1)
    
    # First, find all problem statements with point values
    problem_texts = []
    
    # Find problems that look like: \bigskip {\bf 1. (22 points total)}
    pattern = r'\\bigskip\s*{\\bf\s*(\d+)\.\s*\((\d+)\s*points.*?\)}(.*?)(?=\\bigskip|\\end{document})'
    matches = re.findall(pattern, content, re.DOTALL)
    
    if matches:
        for num, points, text in matches:
            # Skip problem 0 if it exists (usually just affirmation)
            if num == '0':
                continue
                
            # Clean the text by removing LaTeX commands
            cleaned = text.replace('\\Hspace', ' ')
            cleaned = re.sub(r'\\vfill+', '', cleaned)
            cleaned = re.sub(r'\\item\[(.*?)\]', r'- \1', cleaned)
            cleaned = re.sub(r'\\item', '- ', cleaned)
            
            problem_texts.append(f"Problem {num} ({points} points):\n{cleaned}")
    
    # If we didn't find any problems with our regex, extract the whole content
    if not problem_texts:
        # Clean up the whole content
        cleaned_content = content.replace('\\Hspace', ' ')
        cleaned_content = re.sub(r'\\vfill+', '', cleaned_content)
        cleaned_content = re.sub(r'\\bigskip', '\n\n', cleaned_content)
        cleaned_content = re.sub(r'\\medskip', '\n', cleaned_content)
        
        # Extract just the core problem description
        algorithm_problem = "Graduate-level Algorithm Problem Set:\n\n"
        algorithm_problem += "This appears to be a problem set covering flow networks, "
        algorithm_problem += "including residual networks, augmenting paths, "
        algorithm_problem += "minimum-capacity cuts, and the Ford-Fulkerson and Edmonds-Karp algorithms."
        
        return algorithm_problem
    
    # Join all problem texts with separators
    result = "\n\n---\n\n".join(problem_texts)
    return result

# Function to load or create a reference solution
def get_reference_solution(problem_id, problem_statement, use_multimodal=False):
    ref_solution_path = os.path.join("reference_solutions", f"{problem_id}.json")
    
    # Check if reference solution already exists
    if os.path.exists(ref_solution_path):
        print(f"Loading existing reference solution for problem {problem_id}...")
        with open(ref_solution_path, "r") as f:
            return json.load(f)["reference_solution"]
    
    # Generate new reference solution
    print(f"Generating new reference solution for problem {problem_id}...")
    temp_state = {
        "pdf_path": "",
        "student_id": "reference",
        "problem_statement": problem_statement,
        "structured_text": "",
        "reference_solution": "",
        "llm_grading": "",
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    updated_state = generate_reference_solution_reasoning(temp_state)
    
    # Save the reference solution
    with open(ref_solution_path, "w") as f:
        json.dump({"problem_id": problem_id, 
                  "problem_statement": problem_statement,
                  "reference_solution": updated_state["reference_solution"],
                  "timestamp": updated_state["timestamp"]}, f, indent=4)
    
    return updated_state["reference_solution"]

# Main function for running the grading pipeline
def main():
    parser = argparse.ArgumentParser(description="LLM-based Automatic Grader")
    parser.add_argument("--problem_id", type=str, help="Unique identifier for the problem set")
    parser.add_argument("--multimodal", action="store_true", help="Use multimodal LLM for reference solution")
    parser.add_argument("--tex_file", type=str, help="Path to the LaTeX problem statement file")
    parser.add_argument("--text_file", type=str, help="Path to the text file containing the problem statement")
    parser.add_argument("--ocr", type=str, choices=["gpt4o", "mistral"], default="gpt4o", 
                       help="OCR provider to use (gpt4o or mistral)")
    args = parser.parse_args()
    
    # Get problem details
    if not args.problem_id:
        problem_id = input("Enter problem ID (e.g., 'hw1_problem2'): ")
    else:
        problem_id = args.problem_id
    
    # Get or prompt for problem statement
    problem_file = os.path.join("reference_solutions", f"{problem_id}.json")
    
    if os.path.exists(problem_file):
        # Load existing problem statement
        with open(problem_file, "r") as f:
            problem_data = json.load(f)
            problem_statement = problem_data["problem_statement"]
            print(f"Loaded problem statement for {problem_id}")
    else:
        # Try to get problem statement from a text file first
        if args.text_file and os.path.exists(args.text_file):
            try:
                print(f"Reading problem statement from {args.text_file}...")
                with open(args.text_file, 'r') as f:
                    problem_statement = f.read().strip()
                print("Problem statement loaded from text file.")
            except Exception as e:
                print(f"Error reading text file: {str(e)}")
                problem_statement = None
        # If not available or failed, try using TeX file
        elif args.tex_file and os.path.exists(args.tex_file):
            print(f"Extracting problem statement from {args.tex_file}...")
            problem_statement = extract_problem_from_tex(args.tex_file)
            print("Problem statement extracted from TeX file.")
        else:
            # Otherwise prompt the user for the text file path first
            text_path = input("Enter path to problem statement text file (or press Enter to try other options): ")
            
            if text_path and os.path.exists(text_path):
                try:
                    print(f"Reading problem statement from {text_path}...")
                    with open(text_path, 'r') as f:
                        problem_statement = f.read().strip()
                    print("Problem statement loaded from text file.")
                except Exception as e:
                    print(f"Error reading file: {str(e)}")
                    problem_statement = None
            else:
                # If text file approach didn't work, try TeX file
                tex_path = input("Enter path to problem statement TeX file (or press Enter to input manually): ")
                
                if tex_path and os.path.exists(tex_path):
                    problem_statement = extract_problem_from_tex(tex_path)
                    print("Problem statement extracted from TeX file.")
                else:
                    problem_statement = None
                    
            # Fall back to manual input if all automated approaches failed
            if not problem_statement:
                print("Please provide the problem statement:")
                problem_statement = input("> ")
                while not problem_statement.strip():
                    print("Problem statement cannot be empty. Please try again:")
                    problem_statement = input("> ")
    
    # Get reference solution
    reference_solution = get_reference_solution(problem_id, problem_statement, args.multimodal)
    print("Reference solution ready.")
    
    # Process student submissions
    while True:
        # Get student information
        student_id = input("\nEnter student ID (or 'q' to quit): ")
        if student_id.lower() == 'q':
            break
        
        pdf_path = input("Enter path to student solution PDF: ")
        if not os.path.exists(pdf_path):
            print(f"Error: File {pdf_path} does not exist. Please try again.")
            continue
        
        # Make a copy of the student's PDF in our storage
        student_pdf_copy = os.path.join("student_solutions", f"{student_id}_{problem_id}_{int(time.time())}.pdf")
        with open(pdf_path, "rb") as src, open(student_pdf_copy, "wb") as dst:
            dst.write(src.read())
        
        # Prepare state for this student
        initial_state = {
            "pdf_path": pdf_path,
            "student_id": student_id,
            "problem_statement": problem_statement,
            "reference_solution": reference_solution,
            "structured_text": "",
            "llm_grading": "",
            "ocr_provider": args.ocr,  # Use the OCR provider from command line
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Process with LangGraph - start from OCR since we already have reference solution
        try:
            print(f"\nProcessing submission for student {student_id} using {args.ocr} OCR...")
            ocr_state = ocr_step(initial_state)
            graded_state = llm_grading_step(ocr_state)
            final_state = human_verification_step(graded_state)
            print(f"Completed grading for student {student_id}")
        except Exception as e:
            print(f"Error processing student {student_id}: {str(e)}")
    
    print("\nAll student submissions processed. Exiting.")


if __name__ == "__main__":
    main()
