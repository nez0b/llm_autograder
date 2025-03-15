import re
import os
import json
import datetime
from pathlib import Path

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

def mock_generate_reference_solution(problem_statement):
    """Mock function to simulate generating a reference solution"""
    return f"""Here's a reference solution for the problem:

1. Constructing the residual network in O(|E|+|V|) time:
   - For each edge (u,v) with capacity c and flow f in G, add an edge in G_f
   - If f < c, add edge (u,v) with capacity c-f in G_f
   - If f > 0, add edge (v,u) with capacity f in G_f
   - This requires iterating through all edges once, thus O(|E|) time
   - Creating the vertex set is O(|V|)
   - Total time: O(|E|+|V|)

2. Determining augmenting paths in O(|E|+|V|) time:
   - Run BFS or DFS from source s to sink t in G_f
   - BFS/DFS has time complexity O(|E|+|V|)
   - If t is reachable from s, an augmenting path exists

3. Proof that f is a maximum flow if no augmenting path exists:
   - If no augmenting path exists, s and t are disconnected in G_f
   - This creates an s-t cut (S,T) where S is the set of vertices reachable from s in G_f
   - For any edge (u,v) where u ∈ S and v ∈ T, the edge must be saturated in G
   - For any edge (v,u) where u ∈ S and v ∈ T, the flow must be 0
   - Therefore, the flow across this cut equals its capacity
   - By the max-flow min-cut theorem, f is a maximum flow

4. Finding minimum-capacity cut in O(|E|+|V|) time:
   - Run BFS/DFS from s in G_f to find set S of reachable vertices
   - Set T = V - S
   - (S,T) is a minimum capacity cut
   - Time complexity is O(|E|+|V|) for the BFS/DFS"""

def mock_ocr_process(pdf_path):
    """Mock function to simulate OCR extraction from PDF"""
    return """# Quiz 4 Solution

AFFIRMED: John Smith (js123456)

## Problem 1

(a) To construct the residual network G_f in O(|E|+|V|) time:
- Iterate through each edge (u,v) in G with capacity c(u,v) and flow f(u,v)
- For each edge, add forward edge (u,v) with capacity c(u,v)-f(u,v) if c(u,v)-f(u,v) > 0
- Add backward edge (v,u) with capacity f(u,v) if f(u,v) > 0
- This requires one pass through all edges, taking O(|E|) time
- Creating the vertex set takes O(|V|) time
- Total complexity: O(|E|+|V|)

(b) To determine if an augmenting path exists in G_f in O(|E|+|V|) time:
- Run BFS from source s to sink t in G_f
- BFS has time complexity O(|E|+|V|)
- If t is reached, an augmenting path exists; otherwise, there is none

(c) If there is no augmenting path in G_f, then f is a maximum flow:
- Let S be the set of vertices reachable from s in G_f
- Let T = V - S (vertices not reachable from s)
- (S,T) forms a cut in G
- For any edge (u,v) with u ∈ S and v ∈ T, we must have f(u,v) = c(u,v)
- For any edge (v,u) with u ∈ S and v ∈ T, we must have f(v,u) = 0
- The value of flow f equals the capacity of cut (S,T)
- By max-flow min-cut theorem, f is a maximum flow

(d) To find a minimum-capacity cut if no augmenting path exists:
- Run BFS/DFS from s in G_f
- S = set of vertices reachable from s
- T = V - S
- (S,T) is a minimum-capacity cut
- This takes O(|E|+|V|) time for the BFS/DFS traversal"""

def mock_grade_solution(problem_statement, student_solution, reference_solution):
    """Mock function to simulate LLM grading"""
    return """## Grading Summary

### Problem Understanding: 2/2 points
The student demonstrates excellent understanding of the problem. They correctly identified that the questions involve constructing residual networks, finding augmenting paths, proving maximum flow properties, and finding minimum-capacity cuts.

### Algorithm Design: 3/3 points
The student's algorithms are correct and efficient:
- Their approach to constructing the residual network in O(|E|+|V|) time is correct
- Their BFS approach to finding augmenting paths is optimal
- Their method of finding the minimum-capacity cut is correct

### Correctness: 3/3 points
The student's solutions are mathematically sound:
- Their construction of the residual network accounts for both forward and backward edges
- Their proof that no augmenting path implies maximum flow is complete
- Their explanation of finding the minimum-capacity cut is accurate

### Analysis & Proofs: 2/2 points
The student provides excellent analysis:
- Time complexity analysis is correct for all parts
- The proof of maximum flow using the max-flow min-cut theorem is well-explained
- They correctly identify how to construct the minimum cut

### Total Score: 10/10

### Areas for Improvement
While the solution is excellent, the student could:
- Include a brief example to illustrate the construction of a residual network
- Elaborate on how to implement the BFS algorithm specifically for this problem"""

def main():
    # Create storage directories
    os.makedirs("reference_solutions", exist_ok=True)
    os.makedirs("graded_results", exist_ok=True)
    os.makedirs("student_solutions", exist_ok=True)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Test the grading workflow")
    parser.add_argument("--ocr", type=str, choices=["gpt4o", "mistral"], default="gpt4o", 
                         help="OCR provider to use (gpt4o or mistral)")
    args = parser.parse_args()
    
    problem_id = "quiz4"
    tex_file_path = "/Users/bozen/Code/Agents/grader/Example_Problem.tex"
    student_pdf_path = "/Users/bozen/Code/Agents/grader/example_student_sol.pdf"
    ocr_provider = args.ocr
    
    # 1. Extract problem from TeX file
    print(f"\n--- Step 1: Extracting problem statement from {tex_file_path} ---")
    problem_statement = extract_problem_from_tex(tex_file_path)
    print(problem_statement[:200] + "..." if len(problem_statement) > 200 else problem_statement)
    
    # 2. Generate or load reference solution
    print("\n--- Step 2: Generating reference solution ---")
    reference_solution = mock_generate_reference_solution(problem_statement)
    print(reference_solution[:200] + "..." if len(reference_solution) > 200 else reference_solution)
    
    # Save reference solution
    ref_solution_path = os.path.join("reference_solutions", f"{problem_id}.json")
    with open(ref_solution_path, "w") as f:
        json.dump({
            "problem_id": problem_id,
            "problem_statement": problem_statement,
            "reference_solution": reference_solution,
            "timestamp": datetime.datetime.now().isoformat()
        }, f, indent=4)
    print(f"Reference solution saved to {ref_solution_path}")
    
    # 3. Process student submission
    student_id = "test_student"
    
    print(f"\n--- Step 3: Processing student submission for {student_id} ---")
    print(f"PDF path: {student_pdf_path}")
    
    # OCR the student solution
    print(f"\n--- Step 4: Extracting text from student PDF using {ocr_provider} OCR ---")
    student_solution = mock_ocr_process(student_pdf_path)
    print(student_solution[:200] + "..." if len(student_solution) > 200 else student_solution)
    
    # Grade the solution
    print("\n--- Step 5: Grading student solution ---")
    grading_result = mock_grade_solution(problem_statement, student_solution, reference_solution)
    print(grading_result[:200] + "..." if len(grading_result) > 200 else grading_result)
    
    # Save the grading result
    result_path = os.path.join("graded_results", f"{student_id}_{problem_id}_result.json")
    with open(result_path, "w") as f:
        json.dump({
            "student_id": student_id,
            "problem_id": problem_id,
            "problem_statement": problem_statement,
            "student_solution": student_solution,
            "reference_solution": reference_solution,
            "grading_result": grading_result,
            "timestamp": datetime.datetime.now().isoformat()
        }, f, indent=4)
    
    print(f"\nGrading result saved to {result_path}")
    print("\nWorkflow completed successfully! The actual implementation would:")
    print("1. Use OpenAI o1 with reasoning for reference solution generation")
    print(f"2. Use {ocr_provider.upper()} for OCR on student PDFs")
    print("3. Use LLM grading with detailed rubric")
    print("4. Prompt for human verification of grading results")

if __name__ == "__main__":
    main()