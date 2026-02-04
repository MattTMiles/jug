import os

def debug_par():
    if not os.path.exists('new.par'):
        print("new.par does not exist!")
        return

    print(f"new.par exists, size: {os.path.getsize('new.par')}")
    
    with open('new.par', 'r') as f:
        lines = f.readlines()
        
    print(f"Read {len(lines)} lines.")
    
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if not parts: continue
        
        if parts[0].startswith('FB'):
            print(f"Line {i}: {repr(line)}")
            print(f"  Parts: {parts}")
            try:
                val = float(parts[1])
                print(f"  Parsed val: {val}")
            except Exception as e:
                print(f"  Error parsing val: {e}")

if __name__ == "__main__":
    debug_par()
