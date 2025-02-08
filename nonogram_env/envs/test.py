import numpy as np

np.random.seed(42)
mtx = np.random.randint(0, 2, size=(10, 10))

def extract_clues(size, matrix):
    row_clues = []

    for row in matrix:
        clue = []
        count = 0  
        
        for col in range(size):
            if row[col] == 1:
                count += 1 
            else:
                if count > 0:  
                    clue.append(count)
                    count = 0  
        
        if count > 0: 
            clue.append(count)

        # Them kich thuoc
        while len(clue) < size:
            clue.append(0)

        row_clues.append(clue)

    return row_clues

print(mtx)
r_clues = extract_clues(10, mtx)
c_clues = extract_clues(10, mtx.T)
print(r_clues)
print(c_clues)