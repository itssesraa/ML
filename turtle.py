import time

# عداد العقد
nodes_expanded = 0

# ----------------------------
# PRINT BOARD
# ----------------------------
def print_board(board):
    for row in board:
        print(row)

# ----------------------------
# FIND EMPTY CELL
# ----------------------------
def find_empty(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return i, j
    return None

# ----------------------------
# CHECK VALID MOVE
# ----------------------------
def is_valid(board, num, pos):
    row, col = pos

    # Check row
    for j in range(9):
        if board[row][j] == num and j != col:
            return False

    # Check column
    for i in range(9):
        if board[i][col] == num and i != row:
            return False

    # Check 3x3 box
    box_x = col // 3
    box_y = row // 3

    for i in range(box_y * 3, box_y * 3 + 3):
        for j in range(box_x * 3, box_x * 3 + 3):
            if board[i][j] == num and (i, j) != pos:
                return False

    return True

# ----------------------------
# DFS + BACKTRACKING SOLVER
# ----------------------------
def solve(board):
    global nodes_expanded

    empty = find_empty(board)
    if not empty:
        return True

    row, col = empty

    for num in range(1, 10):
        nodes_expanded += 1

        if is_valid(board, num, (row, col)):
            board[row][col] = num

            if solve(board):
                return True

            board[row][col] = 0   # backtrack

    return False

# ----------------------------
# RUN TEST
# ----------------------------
def run_sudoku(board, level):
    global nodes_expanded
    nodes_expanded = 0

    print("\n============================")
    print(level, "SUDOKU BEFORE:\n")
    print_board(board)

    start_time = time.time()

    solved = solve(board)

    end_time = time.time()

    print("\n" + level + " SUDOKU AFTER:\n")
    print_board(board)

    print("\n---", level, "ANALYSIS ---")
    print("Solution Found:", solved)
    print("Time taken:", round(end_time - start_time, 5), "seconds")
    print("Nodes Expanded:", nodes_expanded)


# ----------------------------
# EASY PUZZLE
# ----------------------------
easy_board = [
    [5,3,0,0,7,0,0,0,0],
    [6,0,0,1,9,5,0,0,0],
    [0,9,8,0,0,0,0,6,0],
    [8,0,0,0,6,0,0,0,3],
    [4,0,0,8,0,3,0,0,1],
    [7,0,0,0,2,0,0,0,6],
    [0,6,0,0,0,0,2,8,0],
    [0,0,0,4,1,9,0,0,5],
    [0,0,0,0,8,0,0,7,9]
]

# ----------------------------
# HARD PUZZLE
# ----------------------------
hard_board = [
    [0,0,0,0,0,0,0,1,2],
    [0,0,0,0,0,0,0,0,0],
    [0,0,1,0,9,5,0,0,0],
    [0,0,0,0,0,0,3,0,0],
    [0,0,0,8,0,3,0,0,0],
    [0,0,0,0,2,0,0,0,0],
    [0,0,0,0,0,0,2,8,0],
    [0,0,0,4,1,9,0,0,0],
    [0,0,0,0,0,0,0,0,0]
]

# ----------------------------
# RUN BOTH
# ----------------------------
run_sudoku(easy_board, "EASY")
run_sudoku(hard_board, "HARD")