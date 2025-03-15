import os
import re

def extract_problem_from_tex(tex_file_path):
    """Extract the problem statement from a TeX file"""
    with open(tex_file_path, 'r') as f:
        tex_content = f.read()
    
    # Extract the content between document tags
    match = re.search(r'\\begin{document}(.*?)\\end{document}', tex_content, re.DOTALL)
    if match:
        content = match.group(1)
        # Basic cleaning
        content = content.replace('\\Hspace', ' ')
        content = re.sub(r'\\vfill', '', content)
        return content
    else:
        return "Problem content extraction failed."

def main():
    # Get problem statement from TeX file
    problem_tex_path = "/Users/bozen/Code/Agents/grader/Example_Problem.tex"
    problem_statement = extract_problem_from_tex(problem_tex_path)
    print("\n--- Problem Statement ---")
    print(problem_statement[:300] + "..." if len(problem_statement) > 300 else problem_statement)

if __name__ == "__main__":
    main()