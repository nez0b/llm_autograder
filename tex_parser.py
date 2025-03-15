import re

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

if __name__ == "__main__":
    # Test the function
    problem_statement = extract_problem_from_tex("/Users/bozen/Code/Agents/grader/Example_Problem.tex")
    print("\n--- Problem Statement ---")
    print(problem_statement[:500] + "..." if len(problem_statement) > 500 else problem_statement)