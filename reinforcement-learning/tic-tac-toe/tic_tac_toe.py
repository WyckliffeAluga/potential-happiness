# -*- coding: utf-8 -*-
"""
Created on Wed May 13 21:51:20 2020

@author: wyckliffe
"""


import numpy as np

length = 3



class Environment:
    
    def __init__(self): 
        
        self.board = np.zeros((length, length))
        self.x = -1 # represents an x on the board, player 1 
        self.o = 1 # represents an o on the board, player 2 
        self.Winner = None 
        self.ended = False 
        self.number_of_states = 3 ** (length * length)
        
    def is_empty(self, i ,j): 
        
        return self.board[i , j] == 0
    
    def reward(self, sym): 
        
        # no reward until game is over 
        if not self.game_over(): 
            return 0 
        
        # if we get there, game is over 
        # sym will be self.x or self.o
        return 1 if self.winner == sym else 0 
    
    def get_state(self): 
        
        # returns the current state 
        # from 0......|S|-1, where S = set of all possible states 
        # |S| = 3^(Board Sice) since each cell can have 3 possible values - empty, x , o 
        # some states are not possible e.g. all cells are x 
        # this is like finding the interger represented by base-3 number 
        k = 0 
        h = 0 
        
        for i in range(length): 
            for j in range(length): 
                if self.board[i, j] == 0 :
                    v = 0 
                elif self.board[i, j] == self.x :
                    v = 1 
                elif self.board[i, j] == self.o : 
                    v = 2 
                
                h += (3 ** k )  * v 
                k += 1 
        return h 
        
    def game_over(self, force_recalculate=False) : 
        
        if not force_recalculate and self.ended :
            return self.ended 
        
        # check rows
        for i in range(length): 
            for player in (self.x , self.o): 
                if self.board[i].sum() == player * length : 
                    self.winner = player 
                    self.ended = True 
                    return True 
                
        
        # check columns 
        for j in range(length) :
            for player in (self.x , self.o) : 
                if self.board[:,j].sum() == player * length : 
                    self.winner = player 
                    self.ended = True 
                    return True 
                
        # check diagonals 
        for player in (self.x , self.o): 
            
            # top-left -> botto-right diagnoal 
            if self.board.trace() == player * length : 
                self.winner = player 
                self.ended = True 
                return True 
            
            # top-right --> bottom -left diagonal 
            if np.fliplr(self.board).trace() == player * length : 
                self.winner = player 
                self.ended = True 
                return True 
            
        # check if draw 
        if np.all((self.board == 0) == False): 
            # winner stays none 
            self.winner = None 
            self.ended = True 
            return True 
        
        # game is not over 
        self.winner = None 
        return False 
    
    def is_draw(self): 
        
        return self.ended and self.winner is None
    
    def draw_board(self): 
        
        for i in range(length):
            print("-------------")
            for j in range(length):
                print("  ", end="")
                if self.board[i,j] == self.x:
                    print("x ", end="")
                elif self.board[i,j] == self.o:
                    print("o ", end="")
                else:
                    print("  ", end="")
            print("")
        print("-------------")
    
                

class Agent: 
    
    def __init__(self, eps=0.1, alpha=0.5): 
        
        self.eps = eps # probability of choosing ramdnom action instead of greedy 
        self.alpha = alpha # learning rate 
        self.verbose = False 
        self.state_history = [] 
        
    def set_v(self, V): 
        self.V = V 
        
    def set_symbol(self, sym): 
        self.sym = sym 
        
    def set_verbose(self, v): 
        # if true , will print values ofr each position on the board 
        self.verbose = v 
        
    def reset_history(self): 
        self.state_history = [] 
        
    def take_action(self, env): 
        
        # choose an action bases on epsilon-greedy strategy 
        r = np.random.rand() 
        best_state = None 
        
        if r < self.eps : 
            # take a random action 
            if self.verbose : 
                print("Taking a random action ")
                
            possible_moves = [] 
            
            for i in range(length): 
                for j in range(length): 
                    if env.is_empty(i, j): 
                        possible_moves.append((i,j))
            
            idx = np.random.choice(len(possible_moves))
            next_move = possible_moves[idx]
            
        else: 
            # choose actions based on current values of states
            # loop through all possible moves, get their values 
            # keep trakc of the best value
            pos2value = {}
            next_move = None 
            best_value = -1 
            
            for i in range(length): 
                for j in range(length): 
                    
                    if env.is_empty(i ,j): 
                        # what is the state if we make this move 
                        env.board[i,j] = self.sym 
                        state = env.get_state() 
                        env.board[i,j] = 0  # change the board back 
                        pos2value[(i,j)] = self.V[state]
                        
                        if self.V[state] > best_value : 
                            best_value = self.V[state]
                            best_state = state 
                            next_move = (i, j)
                        
            # verbose, draw board with values 
            if self.verbose : 
                print("Taking a greedy action")
                for i in range(length):
                  print("------------------")
                  for j in range(length):
                    if env.is_empty(i, j):
                      # print the value
                      print(" %.2f|" % pos2value[(i,j)], end="")
                    else:
                      print("  ", end="")
                      if env.board[i,j] == env.x:
                        print("x  |", end="")
                      elif env.board[i,j] == env.o:
                        print("o  |", end="")
                      else:
                        print("   |", end="")
                  print("")
                print("------------------")
      
        # make the move
        env.board[next_move[0], next_move[1]] = self.sym
    
    def update_state_history(self, s): 
        
        self.state_history.append(s)
    
    def update(self, env): 
        
        # we ned to backtrack over the states so that 
        # V(prev) = V(prev) + alpha(V(next)state) - V(prev_state)
        # V(next_state) = reward if its the most current state 
        
        reward = env.reward(self.sym)
        target = reward 
        
        for prev in reversed(self.state_history): 
            value = self.V[prev] + self.alpha*(target - self.V[prev])
            self.V[prev] = value 
            target = value 
        
        self.reset_history()
        
class Human: 
    
    def __init__(self) : 
        pass 
    
    def set_symbol(self, sym) : 
        self.sym = sym 
        
    def take_action(self, env): 
        
        while True :
            # break if we maek a legal movve 
            move = input("Enter cordinates 1 , j for your next move (i, j = 0..2:") 
            
            i, j = move.split(',')
            i = int(i)
            j = int(j)
            
            if env.is_empty(i, j): 
                env.board[i,j] = self.sym 
                break 
        
    def update(self, env): 
        pass 
    
    def update_state_history(self, s): 
        pass 
    

def play_game(p1, p2, env, draw=False): 
    # loops until the game is over 
    current_player = None 
    
    while not env.game_over(): 
        
        # alternate between players 
        # p1 alwasy starts first 
        if current_player == p1 : 
            current_player = p2 
        
        else:
            current_player = p1 
        
        # draw the board before the use who wants to see it makes a move 
        if draw: 
            
            if draw == 1 and current_player == p1 : 
                env.draw_board() 
            if draw == 2 and current_player == p2 : 
                env.draw_board() 
                
       
        
        # current player makes a move 
        current_player.take_action(env)
       
        # update state histories 
        state = env.get_state() 
        p1.update_state_history(state)
        p2.update_state_history(state)
       
    if draw: 
        env.draw_board() 
        
    # do value function update 
    p1.update(env)
    p2.update(env)

def get_state_harsh_and_winner(env, i=0, j=0): 
    
    results = [] 
    
    for v in (0, env.x, env.o): 
        
        env.board[i,j] = v # if empty board it should already be 0 
        
        if j == 2: 
            # j goes back to 0, increases i , unless i = 3, then we are done 
            if i == 2: 
                # the board is full ,coolect resutls and return 
                state = env.get_state()
                ended = env.game_over(force_recalculate=True)
                winner = env.winner
                results.append((state, winner, ended))
            
            else : 
                results += get_state_harsh_and_winner(env, i + 1, 0)
        
        else: 
            # increment j, i stays the same 
            results += get_state_harsh_and_winner(env, i, j + 1)
            
    return results 
        

def initialV_x(env, state_winner_triples): 
    
    # initialize state values as follows 
    # if x wins V(s) = 1 
    # if x loses or draw, V(s) = 0 
    # otherwise, V(s) = 0.5 
    
    V = np.zeros(env.number_of_states)
    
    for state, winner , ended in state_winner_triples: 
        if ended: 
            if winner == env.x :
                v = 1
            else: 
                v = 0 
        else : 
            v = 0.5 
        
        V[state] = v 
    
    return V

def initialV_o(env, state_winner_triples) : 
    
    # this is (almost) the opposite of inital V for player x 
    # sinve everywhere where x wins , o loses 
    # but a draw is still 0 for o 
    V = np.zeros(env.number_of_states)
    
    for state, winner, ended in state_winner_triples : 
        if ended: 
            if winner == env.o : 
                v = 1 
            else: 
                v = 0 
        else: 
            v = 0.5 
        V[state] = v 
    
    return V 


if __name__ == "__main__"    : 
    
    # train the agent 
    p1 = Agent() 
    p2 = Agent()  
    
    # set up environment 
    
    env = Environment() 
    state_winner_triples = get_state_harsh_and_winner(env)
    
    Vx = initialV_x(env, state_winner_triples)
    p1.set_v(Vx)
    
    Vo = initialV_o(env, state_winner_triples)
    p2.set_v(Vo)
    
      # give each player their symbol
    p1.set_symbol(env.x)
    p2.set_symbol(env.o)

    T = 10000
    for t in range(T):
        if t % 200 == 0:
            print(t)
        play_game(p1, p2, Environment())

      # play human vs. agent
      # do you think the agent learned to play the game well?
    human = Human()
    human.set_symbol(env.o)
    while True:
        p1.set_verbose(True)
        play_game(p1, human, Environment(), draw=2)
        
        answer = input("Play again? [Y/n]: ")
        if answer and answer.lower()[0] == 'n':
            break

    
        