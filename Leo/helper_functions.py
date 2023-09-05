# Function to print text within a box
def print_boxed_text(lines, is_title=None):
    if not lines:
        print("No lines to print.")
        return
    
    if is_title is None:
        is_title = [False] * len(lines)
    
    max_length = max(len(line) for line in lines)
    
    print("┌" + "─" * (max_length + 2) + "┐")
    
    for line, title in zip(lines, is_title):
        if title:
            print("├" + "─" * (max_length + 2) + "┤")
            print("│ " + line.center(max_length, ' ') + " │")
            print("├" + "─" * (max_length + 2) + "┤")
            print("│ " + " " * max_length + " │")  # Empty line under the title
        else:
            print("│ " + line + " " * (max_length - len(line)) + " │")
            
    print("└" + "─" * (max_length + 2) + "┘")

# Function to save text within a box to a file (used for when the text is too long to print)
def save_boxed_text_to_file(lines, is_title=None, filename="boxed_text.txt"):
    if not lines:
        return
    
    if is_title is None:
        is_title = [False] * len(lines)
    
    max_length = max(len(line) for line in lines)
    
    with open(filename, 'w') as file:
        file.write("┌" + "─" * (max_length + 2) + "┐\n")
        
        for line, title in zip(lines, is_title):
            if title:
                file.write("├" + "─" * (max_length + 2) + "┤\n")
                file.write("│ " + line.center(max_length, ' ') + " │\n")
                file.write("├" + "─" * (max_length + 2) + "┤\n")
                file.write("│ " + " " * max_length + " │\n")  # Empty line under the title
            else:
                file.write("│ " + line + " " * (max_length - len(line)) + " │\n")
                
        file.write("└" + "─" * (max_length + 2) + "┘\n")
