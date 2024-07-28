# Tic-Tac-Toe with AI and Two-Player Mode

This is a Tic-Tac-Toe game built with Streamlit that features two exciting modes:
1. **Play Against the Agent**: An AI agent trained using the Q-learning algorithm.
2. **Two-Player Mode**: A classic Tic-Tac-Toe game for two human players.

## Features
- **Q-learning Agent**: The AI agent learns and improves its gameplay through Q-learning.
- **Two-Player Mode**: Play against a friend in a traditional Tic-Tac-Toe setup.
- **Score Tracking**: Keeps track of wins, losses, and draws.
- **Invalid Move Handling**: Alerts the player if they attempt to make an invalid move.
- **Game Reset**: Easily reset the game to play again without losing the score.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/omarelnabawi/tic-tac-toe-ai.git
    cd tic-tac-toe-ai
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## How to Play

### Play Against the Agent
1. Select **Play against agent** mode from the sidebar.
2. Enter your move (1-9) and click **Submit Move**.
3. The agent will automatically make its move.
4. The game will continue until there's a winner or a draw.
5. Click **Reset Game** to start a new game.

### Two-Player Mode
1. Select **Two players** mode from the sidebar.
2. Players take turns entering their moves (1-9) and clicking **Submit Move**.
3. The game will continue until there's a winner or a draw.
4. Click **Reset Game** to start a new game.

## About
This project is part of my internship at CodSoft for AI. The goal was to create an engaging AI-powered game using the Q-learning algorithm for the agent and a traditional two-player mode.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Streamlit documentation and community for their support and resources.
- CodSoft for the internship opportunity and guidance.
