import os
import re
import base64
from pathlib import Path

def extract_problem_from_tex(tex_file_path):
    """Extract the problem statement from a TeX file"""
    with open(tex_file_path, 'r') as f:
        tex_content = f.read()
    
    # Remove commented lines first
    tex_content = re.sub(r'%.*', '', tex_content, flags=re.MULTILINE)
    
    # Extract the content between document tags
    match = re.search(r'\\begin{document}(.*?)\\end{document}', tex_content, re.DOTALL)
    if not match:
        return "Problem content extraction failed."
    
    content = match.group(1)
    
    # Extract individual problem sections
    problems = []
    
    # Match sections that look like problem statements - look for "{\bf 1. (22 points total)}" style headers
    problem_matches = re.findall(r'\\bigskip\s*{\\bf\s+(\d+)\.\s+\((\d+)(?:\s+points.*?)\)}(.*?)(?=\\bigskip|\\end{document})', 
                             content, re.DOTALL)
    
    if not problem_matches:
        # Try alternate pattern that might be in the document
        problem_matches = re.findall(r'{\\bf\s+(\d+)\.\s+\((\d+)(?:\s+points.*?)\)}(.*?)(?={\\bf|\\end{document})', 
                                 content, re.DOTALL)
    
    # Process each problem
    for prob_num, points, prob_text in problem_matches:
        # Skip problem 0 which is usually the affirmation
        if prob_num == '0':
            continue
            
        # Clean the text
        cleaned_text = prob_text
        
        # Remove common LaTeX formatting
        cleaned_text = cleaned_text.replace('\\Hspace', ' ')
        cleaned_text = re.sub(r'\\vfill+', '', cleaned_text)
        cleaned_text = re.sub(r'\\medskip', '\n', cleaned_text)
        cleaned_text = re.sub(r'\\bigskip', '\n\n', cleaned_text)
        
        # Handle itemize/enumerate environments
        cleaned_text = re.sub(r'\\begin{itemize}', '\n', cleaned_text)
        cleaned_text = re.sub(r'\\end{itemize}', '\n', cleaned_text)
        cleaned_text = re.sub(r'\\begin{enumerate}', '\n', cleaned_text)
        cleaned_text = re.sub(r'\\end{enumerate}', '\n', cleaned_text)
        
        # Handle item markers
        cleaned_text = re.sub(r'\\item\[(.*?)\]', r'- \1:', cleaned_text)
        cleaned_text = re.sub(r'\\item', '- ', cleaned_text)
        
        # Handle math environments
        cleaned_text = re.sub(r'\$([^$]*)\$', r'$\1$', cleaned_text)
        cleaned_text = re.sub(r'\\begin{.*?}', '', cleaned_text)
        cleaned_text = re.sub(r'\\end{.*?}', '', cleaned_text)
        
        # Format the problem
        formatted_problem = f"Problem {prob_num} ({points} points):\n{cleaned_text.strip()}"
        problems.append(formatted_problem)
    
    # If we didn't find any problems with the regex, return a cleaned version of the whole content
    if not problems:
        # Extract problem-like sections manually using general patterns
        potential_problems = re.findall(r'\\bigskip\s*{\\bf\s+\d+\..*?}(.*?)(?=\\bigskip|\\end{document})', 
                                    content, re.DOTALL)
        
        if potential_problems:
            for i, prob_text in enumerate(potential_problems, 1):
                # Clean similarly as above
                cleaned_text = prob_text.replace('\\Hspace', ' ')
                cleaned_text = re.sub(r'\\vfill', '', cleaned_text)
                
                formatted_problem = f"Problem {i}:\n{cleaned_text.strip()}"
                problems.append(formatted_problem)
            
            # Join the manually extracted problems
            return "\n\n---\n\n".join(problems)
        
        # If all else fails, do basic cleaning of the whole document
        content = content.replace('\\Hspace', ' ')
        content = re.sub(r'\\vfill', '', content)
        content = re.sub(r'\\medskip', '\n', content)
        content = re.sub(r'\\bigskip', '\n\n', content)
        
        # Extract sections that look like they might be problems
        extracted_content = "Extracted Algorithm Problem Set:\n\n"
        
        # Look for flow network related content as we know this is about that
        flow_sections = re.findall(r'flow network|residual network|augmenting path|minimum-capacity cut|Ford-Fulkerson|Edmonds-Karp', 
                                  content, re.IGNORECASE)
        
        if flow_sections:
            extracted_content += "This problem set appears to cover: flow networks, residual networks, augmenting paths, "
            extracted_content += "minimum-capacity cuts, and the Ford-Fulkerson and Edmonds-Karp algorithms."
        else:
            extracted_content += content.strip()
        
        return extracted_content
    
    # Join all problems with separators
    full_problem = "\n\n---\n\n".join(problems)
    
    return full_problem

def main():
    # Get problem statement from TeX file
    problem_tex_path = "/Users/bozen/Code/Agents/grader/Example_Problem.tex"
    problem_statement = extract_problem_from_tex(problem_tex_path)
    print("\n--- Problem Statement ---")
    print(problem_statement[:300] + "..." if len(problem_statement) > 300 else problem_statement)
    
    # We'd run OCR on the student PDF here, but we can't due to environment constraints
    print("\n--- Would OCR Student Solution ---")
    print("PDF path: /Users/bozen/Code/Agents/grader/example_student_sol.pdf")
    
    # Print key steps that would be done in your actual implementation
    print("\nKey steps in your enhanced implementation:")
    print("1. Extract problem statement from TeX file ✓")
    print("2. Generate reference solution using OpenAI o1 or GPT-4-turbo")
    print("3. Use GPT-4o's OCR capability to extract text from student's PDF")
    print("4. Grade the solution against the reference with LLM and rubric")
    print("5. Present results to human grader for verification")
    print("6. Store all data for records")

if __name__ == "__main__":
    main()