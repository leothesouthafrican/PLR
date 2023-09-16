from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np

# Used to compute the phi coefficient for the contingency table
def phi_coefficient(contingency_table):
    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        return 0
    a = contingency_table.iloc[0, 0]
    b = contingency_table.iloc[0, 1]
    c = contingency_table.iloc[1, 0]
    d = contingency_table.iloc[1, 1]
    numerator = (a * d) - (b * c)
    denominator = np.sqrt((a + b) * (a + c) * (b + d) * (c + d))
    return numerator / denominator if denominator != 0 else 0

# Used to one-hot encode categorical attributes
def one_hot_encode(df):
    object_cols = df.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_attrs = encoder.fit_transform(df[object_cols])
    
    print("List of one-hot encoded columns:")
    print(list(encoder.get_feature_names_out(object_cols)))

    if len(object_cols) == encoder.n_features_in_:
        encoded_df = pd.DataFrame(encoded_attrs, columns=encoder.get_feature_names_out(object_cols))
    else:
        encoded_df = pd.DataFrame(encoded_attrs, columns=encoder.get_feature_names_out())

    return pd.concat([df.drop(object_cols, axis=1), encoded_df], axis=1)


# Used to standardize numeric attributes
# Used to standardize numeric attributes
def standardize_numeric(df):
    if df.empty:
        print("DataFrame is empty, skipping standardization.")
        return df
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) == 0:
        print("No numeric columns to standardize.")
        return df

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Print the list of standardized columns
    print("List of standardized columns:")
    print(list(numeric_cols))
    
    return df

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
