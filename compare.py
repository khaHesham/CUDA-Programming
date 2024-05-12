import sys

def compare_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    if len(lines1) != len(lines2):
        print("Files have different number of lines. Cannot compare.")
        return

    total_lines = len(lines1)
    matching_lines = 0

    for line1, line2 in zip(lines1, lines2):
        num1 = float(line1.strip())
        num2 = float(line2.strip())
        if num1 == num2:
            matching_lines += 1

    accuracy = (matching_lines / total_lines) * 100
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <file1> <file2>")
        sys.exit(1)

    file1_path = sys.argv[1]
    file2_path = sys.argv[2]

    compare_files(file1_path, file2_path)
