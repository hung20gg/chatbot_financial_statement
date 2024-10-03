def read_file_without_comments(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = [line for line in lines if not line.startswith('#') and not line.startswith('//')]
        return ''.join(lines)
    
    
if __name__ == '__main__':
    print(read_file_without_comments('prompt/seek_database.txt'))