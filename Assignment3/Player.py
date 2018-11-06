#Yibo Fu
#CMPS140 ASG3
import numpy as np

class AIPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)

    def get_alpha_beta_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the alpha-beta pruning algorithm

        This will play against either itself or a human player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        alpha = -float('inf')
        beta = float('inf')
        from datetime import datetime
        start = datetime.now()
        val, action = self.max_value(board, alpha, beta, 6)
        print(datetime.now() - start)
        return action

    def max_value(self, b, alpha, beta, depth):
        if depth <= 0 or self.is_terminal(b) or self.is_oppo_terminal(b):
            return self.evaluation_function(b), None
        action = None
        v = -float('inf')
        for move, successor in self.get_successors(b):
            tmp_v, _ = self.min_value(successor, alpha, beta, depth-1)
            if tmp_v > v:
                action = move
            v = max(v, tmp_v)
            if v >= beta:
                return v, action
            alpha = max(alpha,v)
        return v, action

    def min_value(self, b, alpha, beta, depth):
        if depth <= 0 or self.is_terminal(b)  or self.is_oppo_terminal(b):
            return self.evaluation_function(b), None

        v = float('inf')
        for move, successor in self.get_successors(b):
            tmp_v, _ = self.max_value(successor, alpha, beta, depth-1)
            v = min(v, tmp_v)
            if v <= alpha:
                return v, None
            beta = min(beta, v)
        return v, None

    def get_expectimax_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the expectimax algorithm.

        This will play against the random player, who chooses any valid move
        with equal probability

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        #raise NotImplementedError('Whoops I don\'t know what to do')
        depth = 4
        score, action = self._expectimax_value(board, depth)
        return action

    def _expectimax_value(self, board, depth):
        if self.is_terminal(board) or self.is_oppo_terminal(board) or depth <= 0:
            return self.evaluation_function(board)
        if self.next_agent(board) == self.player_number:
            val, action = self._expectimax_max_value(board, depth-1)
            return val, action
        if self.next_agent(board) != self.player_number:
            return self._expectimax_exp_value(board, depth-1)

    def is_oppo_terminal(self, board):
        player_num = self.opponent()
        player_win_str = '{0}{0}{0}{0}'.format(player_num)
        to_str = lambda a: ''.join(a.astype(str))

        def check_horizontal(b):
            for row in b:
                if player_win_str in to_str(row):
                    return True
            return False

        def check_verticle(b):
            return check_horizontal(b.T)

        def check_diagonal(b):
            for op in [None, np.fliplr]:
                op_board = op(b) if op else b

                root_diag = np.diagonal(op_board, offset=0).astype(np.int)
                if player_win_str in to_str(root_diag):
                    return True

                for i in range(1, b.shape[1] - 3):
                    for offset in [i, -i]:
                        diag = np.diagonal(op_board, offset=offset)
                        diag = to_str(diag.astype(np.int))
                        if player_win_str in diag:
                            return True

            return False

        return (check_horizontal(board) or
                check_verticle(board) or
                check_diagonal(board))

    def is_terminal(self, board):
        player_num = self.player_number
        player_win_str = '{0}{0}{0}{0}'.format(player_num)
        to_str = lambda a: ''.join(a.astype(str))

        def check_horizontal(b):
            for row in b:
                if player_win_str in to_str(row):
                    return True
            return False

        def check_verticle(b):
            return check_horizontal(b.T)

        def check_diagonal(b):
            for op in [None, np.fliplr]:
                op_board = op(b) if op else b

                root_diag = np.diagonal(op_board, offset=0).astype(np.int)
                if player_win_str in to_str(root_diag):
                    return True

                for i in range(1, b.shape[1] - 3):
                    for offset in [i, -i]:
                        diag = np.diagonal(op_board, offset=offset)
                        diag = to_str(diag.astype(np.int))
                        if player_win_str in diag:
                            return True

            return False

        return (check_horizontal(board) or
                check_verticle(board) or
                check_diagonal(board))


    def next_agent(self, board):
        b = board.reshape((1, -1)).tolist()[0]
        player_num = 1 if b.count(1) == b.count(2) else 2
        return player_num

    def _expectimax_max_value(self, board, depth):
        v = -float('inf')
        action = None
        for move, successor in self.get_successors(board):
            score = self._expectimax_value(successor, depth-1)
            if v < score:
                v = score
                action = move
        return v, action

    def _expectimax_exp_value(self, board, depth):
        v = 0
        move_and_successors = self.get_successors(board)
        for move, successor in move_and_successors:
            v += (1/len(move_and_successors)) * self._expectimax_value(successor, depth-1)
        return v

    def get_successors(self, board):
        valid_cols = []
        b = board.reshape((1, -1)).tolist()[0]
        player_num = 1 if b.count(1) == b.count(2) else 2
        for col in range(board.shape[1]):
            if 0 in board[:, col]:
                tmp = board.copy()

                for row in range(1, tmp.shape[0]):
                    update_row = -1
                    if tmp[row, col] > 0 and tmp[row - 1, col] == 0:
                        update_row = row - 1
                    elif row == tmp.shape[0] - 1 and tmp[row, col] == 0:
                        update_row = row

                    if update_row >= 0:
                        tmp[update_row, col] = player_num
                        break
                valid_cols.append([col, tmp])
        
        return valid_cols

    def opponent(self):
        return 2 if self.player_number == 1 else 1


    def evaluation_function(self, board):
        """
        Given the current stat of the board, return the scalar value that
        represents the evaluation function for the current player
       
        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The utility value for the current board
        """

        def check_horizontal(b):
            for row in b:
                if player_win_str in to_str(row):
                    return True
            return False

        def check_verticle(b):
            return check_horizontal(b.T)

        def check_diagonal(b):
            for op in [None, np.fliplr]:
                op_board = op(b) if op else b

                root_diag = np.diagonal(op_board, offset=0).astype(np.int)
                if player_win_str in to_str(root_diag):
                    return True

                for i in range(1, b.shape[1] - 3):
                    for offset in [i, -i]:
                        diag = np.diagonal(op_board, offset=offset)
                        diag = to_str(diag.astype(np.int))
                        if player_win_str in diag:
                            return True

            return False

        if self.is_terminal(board):
            return float('inf')

        v = 0
        player_num = self.player_number
        player_win_str = '{0}{0}{0}{0}'.format(player_num)
        to_str = lambda a: ''.join(a.astype(str))

        v += 10000 if check_diagonal(board) else 0
        v += 10000 if check_verticle(board) else 0
        v += 10000 if check_horizontal(board) else 0

        player_win_str = '{0}{0}{0}'.format(player_num)
        v += 1000 if check_diagonal(board) else 0
        v += 1000 if check_verticle(board) else 0
        v += 1000 if check_horizontal(board) else 0

        player_win_str = '{0}{0}'.format(player_num)
        v += 200 if check_diagonal(board) else 0
        v += 200 if check_verticle(board) else 0
        v += 200 if check_horizontal(board) else 0

        v += self.get_position_score(board, player_num)

        player_win_str = '{0}{0}{0}{0}'.format(self.opponent())
        b = board
        if check_diagonal(b) or check_horizontal(b) or check_verticle(b):
            return -10000
        return v

    def get_position_score(self, board, player_num):
        length = board.shape[1]
        center = length/2
        v = 100
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if board[i][j] == player_num:
                    v -= abs(center-j)
        return v


class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.player_string = 'Player {}:random'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state select a random column from the available
        valid moves.

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:,col]:
                valid_cols.append(col)
        return np.random.choice(valid_cols)


class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {}:human'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state returns the human input for next move

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)

        move = int(input('Enter your move: '))

        while move not in valid_cols:
            print('Column full, choose from:{}'.format(valid_cols))
            move = int(input('Enter your move: '))

        return move

