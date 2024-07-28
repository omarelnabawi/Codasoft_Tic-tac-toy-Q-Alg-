import streamlit as st
import numpy as np
import random
import pickle

class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.array(['1', '2', '3', '4', '5', '6', '7', '8', '9'])
        self.done = False
        self.winner = None
        return self.board

    def available_actions(self):
        return [i for i in range(9) if self.board[i] not in ['X', 'O']]

    def step(self, action, player):
        if self.board[action] in ['X', 'O']:
            return self.board, self.done, self.winner, True
        self.board[action] = 'X' if player == 1 else 'O'
        self.done, self.winner = self.check_winner()
        return self.board, self.done, self.winner, False

    def check_winner(self):
        lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
        for line in lines:
            if self.board[line[0]] == self.board[line[1]] == self.board[line[2]]:
                return True, 1 if self.board[line[0]] == 'X' else -1
        if not self.available_actions():
            return True, 0  # Draw
        return False, None

    def render(self):
        st.write(self.board.reshape(3, 3))
        st.write('-------------------')

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, epsilon_decay=0.99, epsilon_min=0.01):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}

    def get_q_value(self, state, action):
        return self.q_table.get((state.tostring(), action), 0.0)

    def update_q_value(self, state, action, reward, next_state, next_actions):
        old_q = self.get_q_value(state, action)
        future_q = max([self.get_q_value(next_state, a) for a in next_actions], default=0)
        self.q_table[(state.tostring(), action)] = old_q + self.alpha * (reward + self.gamma * future_q - old_q)

    def choose_action(self, state, actions):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(actions)
        q_values = [self.get_q_value(state, action) for action in actions]
        max_q = max(q_values)
        return random.choice([action for action, q_value in zip(actions, q_values) if q_value == max_q])

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent(episodes=50000):
    env = TicTacToe()
    agent = QLearningAgent()

    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            actions = env.available_actions()
            action = agent.choose_action(state, actions)
            next_state, done, winner, _ = env.step(action, 1)
            reward = 1 if winner == 1 else -1 if winner == -1 else 0
            if not done:
                next_actions = env.available_actions()
                agent.update_q_value(state, action, reward, next_state, next_actions)
                state = next_state
            else:
                agent.update_q_value(state, action, reward, next_state, [])

        agent.decay_epsilon()

    with open('q_table.pkl', 'wb') as f:
        pickle.dump(agent.q_table, f)
    return agent

def play_against_agent(agent):
    env = TicTacToe()

    if 'state' not in st.session_state:
        st.session_state.state = env.reset()
        st.session_state.done = False
        st.session_state.winner = None
        st.session_state.player_wins = 0
        st.session_state.agent_wins = 0
        st.session_state.draws = 0
        st.session_state.invalid_move = False

    st.write("Current board:")
    env.board = st.session_state.state
    env.render()

    if st.session_state.done:
      if st.session_state.winner == 1:
          st.sidebar.write("Agent wins!")
          if 'score_updated' not in st.session_state or not st.session_state.score_updated:
              st.session_state.agent_wins += 1
              st.session_state.score_updated = True
      elif st.session_state.winner == -1:
          st.sidebar.write("You win!")
          if 'score_updated' not in st.session_state or not st.session_state.score_updated:
              st.session_state.player_wins += 1
              st.session_state.score_updated = True
      else:
          st.sidebar.write("It's a draw!")
          if 'score_updated' not in st.session_state or not st.session_state.score_updated:
              st.session_state.draws += 1
              st.session_state.score_updated = True

      if st.sidebar.button('Reset Game', key="reset_game_agent"):
          st.session_state.state = env.reset()
          st.session_state.done = False
          st.session_state.winner = None
          st.session_state.history = []
          st.session_state.invalid_move = False
          st.session_state.score_updated = False

    else:
        st.write("Your turn (O)")
        human_action = st.number_input("Enter your move (1-9):", max_value=9, min_value=1, value=1) - 1
        if st.button('Submit Move', key="submit_move_agent"):
            state, done, winner, invalid_move = env.step(human_action, -1)
            if invalid_move:
                st.error("Invalid action, position already taken. Try again.")
                st.session_state.invalid_move = True
            else:
                st.session_state.state = state
                st.session_state.done = done
                st.session_state.winner = winner
                st.session_state.invalid_move = False

                if not done:
                    actions = env.available_actions()
                    action = agent.choose_action(state, actions)
                    state, done, winner, _ = env.step(action, 1)
                    st.session_state.state = state
                    st.session_state.done = done
                    st.session_state.winner = winner

            st.experimental_rerun()

    st.sidebar.write(f"Score: Player: {st.session_state.player_wins}, Agent: {st.session_state.agent_wins}, Draws: {st.session_state.draws}")

def play_two_players():
    env = TicTacToe()

    if 'state_2p' not in st.session_state:
        st.session_state.state_2p = env.reset()
        st.session_state.done_2p = False
        st.session_state.winner_2p = None
        st.session_state.current_player = -1
        st.session_state.player1_wins = 0
        st.session_state.player2_wins = 0
        st.session_state.draws_2p = 0
        st.session_state.invalid_move_2p = False

    st.write("Current board:")
    env.board = st.session_state.state_2p
    env.render()

    if st.session_state.done_2p:
      if st.session_state.winner_2p == 1:
          st.sidebar.write("Player 1 (X) wins!")
          if 'score_updated_2p' not in st.session_state or not st.session_state.score_updated_2p:
              st.session_state.player1_wins += 1
              st.session_state.score_updated_2p = True
      elif st.session_state.winner_2p == -1:
          st.sidebar.write("Player 2 (O) wins!")
          if 'score_updated_2p' not in st.session_state or not st.session_state.score_updated_2p:
              st.session_state.player2_wins += 1
              st.session_state.score_updated_2p = True
      else:
          st.sidebar.write("It's a draw!")
          if 'score_updated_2p' not in st.session_state or not st.session_state.score_updated_2p:
              st.session_state.draws_2p += 1
              st.session_state.score_updated_2p = True

      if st.sidebar.button('Reset Game', key="reset_game_2p"):
          st.session_state.state_2p = env.reset()
          st.session_state.done_2p = False
          st.session_state.winner_2p = None
          st.session_state.current_player = -1
          st.session_state.invalid_move_2p = False
          st.session_state.score_updated_2p = False

    else:
        st.write(f"Player {'1: (X)' if st.session_state.current_player == 1 else '2 (O)'}'s turn")
        human_action = st.number_input("Enter your move (1-9):", max_value=9, min_value=1, value=1, key="move_2p") - 1
        if st.button('Submit Move', key="submit_move_2p"):
            state, done, winner, invalid_move_2p = env.step(human_action, st.session_state.current_player)
            if invalid_move_2p:
                st.error("Invalid action, position already taken. Try again.")
                st.session_state.invalid_move_2p = True
            else:
                st.session_state.state_2p = state
                st.session_state.done_2p = done
                st.session_state.winner_2p = winner
                st.session_state.current_player = 1 if st.session_state.current_player == -1 else -1
                st.session_state.invalid_move_2p = False

            st.experimental_rerun()

    st.sidebar.write(f"Score: Player 1: {st.session_state.player1_wins}, Player 2: {st.session_state.player2_wins}, Draws {st.session_state.draws_2p}")

def main():
    agent = train_agent()

    if 'player_wins' not in st.session_state:
        st.session_state.player_wins = 0
        st.session_state.agent_wins = 0
        st.session_state.draws = 0

    mode = st.sidebar.selectbox("Choose mode:", ['None', 'Play against agent', 'Two players'], key="mode")

    if mode == 'Play against agent':
        play_against_agent(agent)
    elif mode == 'Two players':
        play_two_players()
    elif mode == 'None':
        st.sidebar.write("You must select the mode you want to play in.")

if __name__ == "__main__":
    main()
