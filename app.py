import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
import json
import time

st.set_page_config(
    page_title="Robin Hood Bidding Games",
    page_icon="🏹",
    layout="wide",
)

# Define the core game mechanics
class RobinHoodGame:
    def __init__(self, graph: nx.DiGraph, lambda_param: float, targets: List[str], initial_vertex: str, initial_budget: float):
        self.graph = graph
        self.lambda_param = lambda_param
        self.targets = set(targets)
        self.current_vertex = initial_vertex
        self.player1_budget = initial_budget
        self.player2_budget = 1 - initial_budget
        self.history = []
        self.game_over = False
        self.winner = None
        
        # Record initial state
        self.record_state()
    
    def record_state(self):
        self.history.append({
            'vertex': self.current_vertex,
            'player1_budget': self.player1_budget,
            'player2_budget': self.player2_budget,
        })
    
    def wealth_redistribution(self):
        """Apply the Robin Hood wealth redistribution mechanism"""
        if self.player1_budget > self.player2_budget:
            # Player 1 is richer, pays player 2
            diff = self.player1_budget - self.player2_budget
            transfer = self.lambda_param * diff
            self.player1_budget -= transfer
            self.player2_budget += transfer
        elif self.player2_budget > self.player1_budget:
            # Player 2 is richer, pays player 1
            diff = self.player2_budget - self.player1_budget
            transfer = self.lambda_param * diff
            self.player2_budget -= transfer
            self.player1_budget += transfer
    
    def perform_bidding(self, player1_bid: float, player2_bid: float) -> int:
        """
        Execute the bidding round
        Returns: winner (1 or 2)
        """
        # Ensure bids are valid
        player1_bid = max(0, min(player1_bid, self.player1_budget))
        player2_bid = max(0, min(player2_bid, self.player2_budget))
        
        # Determine winner (player 1 wins ties)
        if player1_bid >= player2_bid:
            winner = 1
            # Player 1 pays bid to player 2
            self.player1_budget -= player1_bid
            self.player2_budget += player1_bid
        else:
            winner = 2
            # Player 2 pays bid to player 1
            self.player2_budget += player2_bid
            self.player1_budget -= player2_bid
            
        return winner
    
    def move_token(self, winner: int, next_vertex: str):
        """Move the token to the next vertex chosen by the winner"""
        if next_vertex not in self.graph.neighbors(self.current_vertex):
            raise ValueError(f"Invalid move. {next_vertex} is not a neighbor of {self.current_vertex}")
        
        self.current_vertex = next_vertex
        
        # Check if game is over
        if self.current_vertex in self.targets:
            self.game_over = True
            self.winner = 1  # Player 1 wins if target is reached
        
        # Record the new state
        self.record_state()
        
    def play_round(self, player1_bid: float, player2_bid: float, next_vertex: str):
        """Play a complete round: WR, bidding, and movement"""
        if self.game_over:
            return
            
        # Apply wealth redistribution
        self.wealth_redistribution()
        
        # Perform bidding and get winner
        winner = self.perform_bidding(player1_bid, player2_bid)
        
        # Move token
        self.move_token(winner, next_vertex)

# UI functions
def create_default_graph():
    """Create the default graph from the paper's example"""
    G = nx.DiGraph()
    G.add_node("vleft")
    G.add_node("vright")
    G.add_node("v1")
    G.add_node("v2")
    
    G.add_edge("vleft", "vright")
    G.add_edge("vright", "vleft")
    G.add_edge("vright", "v1")
    G.add_edge("vleft", "v2")
    
    return G

def display_graph(G, current_vertex=None, targets=None):
    """Display the game graph with highlighted current position and targets"""
    plt.figure(figsize=(8, 6))
    
    pos = nx.spring_layout(G, seed=42)
    
    # Draw regular nodes
    node_colors = ['lightgrey'] * len(G.nodes())
    
    # Highlight targets
    if targets:
        for i, node in enumerate(G.nodes()):
            if node in targets:
                node_colors[i] = 'lightgreen'
    
    # Highlight current vertex
    if current_vertex:
        for i, node in enumerate(G.nodes()):
            if node == current_vertex:
                node_colors[i] = 'orange'
    
    nx.draw(G, pos, with_labels=True, node_color=node_colors, 
            node_size=700, arrowsize=20, font_size=10, font_weight='bold')
    
    # Convert plot to Streamlit
    st.pyplot(plt)

def calculate_threshold_analytically(lambda_param):
    """Calculate threshold for vleft in the example graph from the paper"""
    if lambda_param < 0.25:
        return (2*lambda_param - 2) / (4*lambda_param - 3)
    elif lambda_param == 0.25:
        return 1.0
    else:  # lambda > 0.25
        return 1.0

def display_threshold_graph():
    """Display the threshold as a function of lambda (as in Figure 2 of the paper)"""
    plt.figure(figsize=(8, 5))
    
    lambda_values = np.linspace(0, 0.49, 100)
    threshold_values = [calculate_threshold_analytically(lam) for lam in lambda_values]
    
    plt.plot(lambda_values, threshold_values, 'b-')
    plt.xlabel('λ')
    plt.ylabel('Threshold')
    plt.title('Threshold of vleft as a function of λ')
    plt.ylim(0.5, 1.0)
    plt.grid(True)
    
    st.pyplot(plt)

def display_game_history(game):
    """Display the history of the game as a dataframe"""
    df = pd.DataFrame(game.history)
    
    # Format the budgets to show as percentages
    df['player1_budget'] = df['player1_budget'].apply(lambda x: f"{x:.3f}")
    df['player2_budget'] = df['player2_budget'].apply(lambda x: f"{x:.3f}")
    
    # Add a move number column
    df.insert(0, 'Move', range(len(df)))
    
    st.table(df)

def run_game_simulation(graph, lambda_param, targets, initial_vertex, initial_budget, 
                       player1_strategy, player2_strategy, max_rounds=100):
    """
    Run a complete game simulation with given strategies
    
    Strategies are functions that take the game state and return a bid and next vertex
    """
    game = RobinHoodGame(graph, lambda_param, targets, initial_vertex, initial_budget)
    
    round_num = 0
    while not game.game_over and round_num < max_rounds:
        # Get player 1's move
        p1_bid, p1_next = player1_strategy(game)
        
        # Get player 2's move
        p2_bid, p2_next = player2_strategy(game)
        
        # Perform wealth redistribution and bidding
        game.wealth_redistribution()
        winner = game.perform_bidding(p1_bid, p2_bid)
        
        # Move the token according to winner's choice
        next_vertex = p1_next if winner == 1 else p2_next
        game.move_token(winner, next_vertex)
        
        round_num += 1
        
        # Check for infinite cycle detection
        if round_num >= max_rounds:
            game.game_over = True
            game.winner = 2  # Player 2 wins if no target is reached
    
    return game

# Example strategies
def optimal_player1_strategy(game):
    """
    A simple strategy for player 1 based on the paper's analysis
    """
    current_vertex = game.current_vertex
    neighbors = list(game.graph.neighbors(current_vertex))
    
    # If we can reach a target, go there
    for neighbor in neighbors:
        if neighbor in game.targets:
            # Bid all budget if necessary to reach target
            return game.player1_budget, neighbor
    
    # Otherwise, make a conservative move
    # Simplified threshold calculation for the example graph
    if current_vertex == "vleft":
        # Bid enough to ensure winning to vright
        threshold_diff = 0.1
        return threshold_diff, "vright"
    elif current_vertex == "vright":
        # Try to reach v1
        return game.player1_budget, "v1"
    
    # Fallback - just bid half budget and pick first neighbor
    return game.player1_budget / 2, neighbors[0]

def optimal_player2_strategy(game):
    """
    A simple strategy for player 2 based on the paper's analysis
    """
    current_vertex = game.current_vertex
    neighbors = list(game.graph.neighbors(current_vertex))
    
    # Try to reach v2 from vleft
    if current_vertex == "vleft":
        if "v2" in neighbors:
            return game.player2_budget, "v2"
    
    # Prevent player 1 from reaching targets
    if current_vertex == "vright":
        if "vleft" in neighbors:
            return game.player2_budget, "vleft"
    
    # Fallback - just bid half budget and pick first neighbor that's not a target
    for neighbor in neighbors:
        if neighbor not in game.targets:
            return game.player2_budget / 2, neighbor
    
    # If all neighbors are targets or if there are no neighbors, make a safe move
    if not neighbors:
        # No valid neighbors, this shouldn't happen in normal gameplay
        # Return a safe default that won't be used (since we can't move anywhere)
        return 0, game.current_vertex
    return game.player2_budget / 2, neighbors[0]

# Human input strategy (will be called during interactive play)
def human_player_strategy(game, bid, next_vertex):
    return bid, next_vertex

# Streamlit UI
def main():
    st.title("🏹 Robin Hood Reachability Bidding Games")
    st.markdown("""
    This simulator demonstrates the Robin Hood bidding mechanism introduced in the paper 
    "Robin Hood Reachability Bidding Games" by Shaull Almagor, Guy Avni, and Neta Dafni.
    
    In this model, at the beginning of each round, the richer player pays the poorer player a fraction λ 
    of the difference between their budgets, simulating wealth regulation.
    """)
    
    # How to play guide
    with st.expander("📘 How to Play Guide", expanded=True):
        st.markdown("""
        ### Game Concept
        In Robin Hood Bidding Games, two players compete to move a token on a graph. 
        - **Player 1's goal** is to reach a target vertex (marked in green)
        - **Player 2's goal** is to prevent this by either reaching a safe vertex or keeping the game going indefinitely
        
        ### Game Mechanics
        Each round consists of three phases:
        1. **Wealth Redistribution (Robin Hood mechanism)**: The richer player pays the poorer player a portion (λ) of their wealth difference
        2. **Bidding**: Both players place bids within their budgets, and the highest bidder wins (ties go to Player 1)
        3. **Moving**: The winner moves the token to an adjacent vertex of their choice
        
        ### Available Game Modes
        
        #### 1. Demo from Paper
        - Demonstrates the example from the original research paper
        - Set lambda (λ) and initial budget values
        - Run simulations to see the outcome
        
        #### 2. Interactive Play
        - Play as Player 1 against an AI opponent
        - Make bids and choose moves
        - See the game unfold step by step
        
        #### 3. Theoretical Analysis
        - Explore the mathematical properties behind the game
        - View the threshold function showing how game outcomes change with λ
        
        #### 4. Custom Graph Creation
        - Build your own game graph
        - Add/remove vertices and edges
        - Set target nodes and run simulations
        
        ### Key Concepts
        - **Threshold**: For each vertex, there exists a threshold budget value such that:
          - Player 1 wins if their budget is above the threshold
          - Player 2 wins if their budget is below the threshold
          - At exactly the threshold, the game may be undetermined
        - **Lambda (λ)**: Controls how much wealth redistribution occurs (0 = no redistribution, 0.5 = maximum)
        """)
    
    # Quick start guide for new users
    with st.expander("🚀 Quick Start", expanded=False):
        st.markdown("""
        ### For First-Time Users
        1. Select a game mode from the sidebar on the left
        2. **Demo from Paper**: Just click "Run Simulation" to see the game in action
        3. **Interactive Play**: Make bids and select moves when prompted
        4. **Custom Graph**: Create your own graph before running a simulation
        
        ### Tips for Optimal Play
        - **For Player 1**: 
          - If your budget is above the threshold, try to maintain it while moving toward target
          - Bid conservatively when possible to preserve budget
        - **For Player 2**:
          - Try to move away from target vertices
          - Force Player 1 to spend budget on crucial moves
        """)
    
    
    # Sidebar for configuration
    st.sidebar.header("Game Configuration")
    
    # Select game mode
    game_mode = st.sidebar.radio(
        "Select Game Mode",
        ["Demo from Paper", "Interactive Play", "Theoretical Analysis", "Custom Graph"]
    )
    
    if game_mode == "Demo from Paper":
        st.header("Example Game from the Paper")
        
        # Load the default example graph
        G = create_default_graph()
        
        lambda_param = st.sidebar.slider(
            "Wealth Redistribution Factor (λ)",
            min_value=0.0,
            max_value=0.49,
            value=0.125,
            step=0.025,
            help="The fraction of wealth difference transferred from richer to poorer player"
        )
        
        initial_budget = st.sidebar.slider(
            "Player 1 Initial Budget",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.01
        )
        
        # Display the threshold for this lambda
        threshold = calculate_threshold_analytically(lambda_param)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader(f"Game Graph (λ = {lambda_param})")
            display_graph(G, current_vertex="vleft", targets=["v1"])
            
        with col2:
            st.subheader("Game Parameters")
            st.write(f"Threshold for vleft: **{threshold:.3f}**")
            st.write(f"Player 1 initial budget: **{initial_budget:.3f}**")
            st.write(f"Player 2 initial budget: **{(1-initial_budget):.3f}**")
            
            if initial_budget > threshold:
                st.success("Player 1 has a winning strategy")
            elif initial_budget < threshold:
                st.error("Player 2 has a winning strategy")
            else:
                st.warning("Game may be undetermined exactly at threshold")
        
        # Run a simulation
        if st.button("Run Simulation"):
            with st.spinner("Running simulation..."):
                game = run_game_simulation(
                    G, lambda_param, ["v1"], "vleft", initial_budget,
                    optimal_player1_strategy, optimal_player2_strategy
                )
            
            # Display result
            if game.winner == 1:
                st.success("Player 1 wins by reaching target!")
            else:
                st.error("Player 2 wins by preventing target!")
                
            # Display game history
            st.subheader("Game History")
            display_game_history(game)
    
    elif game_mode == "Interactive Play":
        st.header("Interactive Play")
        
        # Initialize session state for game if not exists
        if 'game' not in st.session_state:
            G = create_default_graph()
            lambda_param = 0.125
            initial_budget = 0.6
            st.session_state.game = RobinHoodGame(G, lambda_param, ["v1"], "vleft", initial_budget)
            st.session_state.game_config = {
                'graph': G,
                'lambda': lambda_param,
                'targets': ["v1"],
                'initial_vertex': "vleft",
                'initial_budget': initial_budget
            }
        
        # Display current game state
        game = st.session_state.game
        config = st.session_state.game_config
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("Game Graph")
            display_graph(game.graph, current_vertex=game.current_vertex, targets=game.targets)
        
        with col2:
            st.subheader("Current State")
            st.write(f"λ = {config['lambda']}")
            st.write(f"Current vertex: **{game.current_vertex}**")
            st.write(f"Player 1 budget: **{game.player1_budget:.3f}**")
            st.write(f"Player 2 budget: **{game.player2_budget:.3f}**")
            
            if game.game_over:
                if game.winner == 1:
                    st.success("Player 1 wins by reaching target!")
                else:
                    st.error("Player 2 wins by preventing target!")
            
        # If game is not over, allow next move
        if not game.game_over:
            st.subheader("Make Your Move (Player 1)")
            
            # Show neighbors
            neighbors = list(game.graph.neighbors(game.current_vertex))
            
            # Player input form
            with st.form("player_move"):
                p1_bid = st.slider(
                    "Your Bid",
                    min_value=0.0,
                    max_value=float(game.player1_budget),
                    value=min(0.1, game.player1_budget),
                    step=0.01
                )
                
                p1_next = st.selectbox("Choose Next Vertex", neighbors)
                
                # For demo purposes, we'll use a simple AI for player 2
                st.write("Player 2 (AI) will respond automatically")
                
                submit_button = st.form_submit_button("Submit Move")
                
                if submit_button:
                    # Get player 2's move using strategy
                    p2_bid, p2_next = optimal_player2_strategy(game)
                    
                    # Display decisions
                    st.write(f"Player 1 bids {p1_bid:.3f} and wants to move to {p1_next}")
                    st.write(f"Player 2 bids {p2_bid:.3f} and wants to move to {p2_next}")
                    
                    # Play round
                    old_vertex = game.current_vertex
                    
                    # Apply wealth redistribution and show the effect
                    old_p1_budget = game.player1_budget
                    old_p2_budget = game.player2_budget
                    
                    game.wealth_redistribution()
                    
                    # Show WR effect
                    st.write("### Wealth Redistribution:")
                    if old_p1_budget > old_p2_budget:
                        transfer = game.lambda_param * (old_p1_budget - old_p2_budget)
                        st.write(f"Player 1 pays {transfer:.3f} to Player 2")
                    elif old_p2_budget > old_p1_budget:
                        transfer = game.lambda_param * (old_p2_budget - old_p1_budget)
                        st.write(f"Player 2 pays {transfer:.3f} to Player 1")
                    
                    # Perform bidding and show winner
                    winner = game.perform_bidding(p1_bid, p2_bid)
                    
                    st.write(f"### Bidding result: Player {winner} wins the bid")
                    
                    # Move according to winner
                    next_vertex = p1_next if winner == 1 else p2_next
                    game.move_token(winner, next_vertex)
                    
                    st.write(f"Token moves from {old_vertex} to {next_vertex}")
                    
                    if game.game_over:
                        if game.winner == 1:
                            st.success("You win by reaching target!")
                        else:
                            st.error("Player 2 wins by preventing target!")
                    
                    # Force refresh
                    st.rerun()
        
        # Display game history
        st.subheader("Game History")
        display_game_history(game)
        
        # Reset button
        if st.button("Reset Game"):
            del st.session_state.game
            del st.session_state.game_config
            st.rerun()
    
    elif game_mode == "Theoretical Analysis":
        st.header("Theoretical Analysis")
        
        st.subheader("Threshold Function")
        st.write("""
        One of the key findings in the paper is that every Robin Hood reachability bidding game has a threshold 
        function. For the example graph, the threshold of vleft as a function of λ is shown below.
        """)
        
        display_threshold_graph()
        
        st.write("""
        The threshold is:
        - For 0 ≤ λ < 1/4: τ(λ) = (2λ-2)/(4λ-3) (increasing from 2/3 to 3/4, 1-strong threshold)
        - For λ = 1/4: τ(λ) = 1 (1-strong threshold)
        - For 1/4 < λ < 1/2: τ(λ) = 1 (2-strong threshold)
        
        Notice the discontinuity at λ = 1/4, which shows an interesting economic interpretation: 
        beyond a certain threshold of redistribution, additional taxation doesn't change the game's 
        fundamental dynamics.
        """)
        
        st.subheader("Game Behavior at Threshold")
        st.write("""
        Interestingly, unlike standard bidding games, Robin Hood games might not be determined exactly at 
        the threshold (i.e., neither player has a winning strategy). This behavior does not occur in 
        standard bidding games.
        """)
        
    elif game_mode == "Custom Graph":
        st.header("Custom Graph Creation")
        
        st.info("This feature allows you to create your own graph to experiment with.")
        
        # Initialize session state for custom graph if not exists
        if 'custom_graph' not in st.session_state:
            st.session_state.custom_graph = create_default_graph()
            st.session_state.custom_nodes = list(st.session_state.custom_graph.nodes())
            st.session_state.custom_edges = list(st.session_state.custom_graph.edges())
            st.session_state.custom_targets = ["v1"]
        
        # Node management
        st.subheader("Manage Nodes")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Add new node
            new_node = st.text_input("Add new node name")
            if st.button("Add Node") and new_node and new_node not in st.session_state.custom_nodes:
                st.session_state.custom_graph.add_node(new_node)
                st.session_state.custom_nodes.append(new_node)
                st.rerun()
        
        with col2:
            # Remove node
            if st.session_state.custom_nodes:
                node_to_remove = st.selectbox("Select node to remove", st.session_state.custom_nodes)
                if st.button("Remove Node"):
                    st.session_state.custom_graph.remove_node(node_to_remove)
                    st.session_state.custom_nodes.remove(node_to_remove)
                    st.session_state.custom_edges = list(st.session_state.custom_graph.edges())
                    
                    # Also remove from targets if present
                    if node_to_remove in st.session_state.custom_targets:
                        st.session_state.custom_targets.remove(node_to_remove)
                    
                    st.experimental_rerun()
        
        # Edge management
        st.subheader("Manage Edges")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Add new edge
            if len(st.session_state.custom_nodes) >= 2:
                source = st.selectbox("Source", st.session_state.custom_nodes, key="source")
                target = st.selectbox("Target", st.session_state.custom_nodes, key="target")
                
                if st.button("Add Edge") and (source, target) not in st.session_state.custom_edges:
                    st.session_state.custom_graph.add_edge(source, target)
                    st.session_state.custom_edges = list(st.session_state.custom_graph.edges())
                    st.experimental_rerun()
        
        with col2:
            # Remove edge
            if st.session_state.custom_edges:
                edge_labels = [f"{s} → {t}" for s, t in st.session_state.custom_edges]
                edge_to_remove = st.selectbox("Select edge to remove", edge_labels)
                
                if st.button("Remove Edge"):
                    idx = edge_labels.index(edge_to_remove)
                    edge = st.session_state.custom_edges[idx]
                    st.session_state.custom_graph.remove_edge(*edge)
                    st.session_state.custom_edges = list(st.session_state.custom_graph.edges())
                    st.experimental_rerun()
        
        # Target management
        st.subheader("Manage Targets")
        
        all_targets = st.multiselect(
            "Select target nodes",
            st.session_state.custom_nodes,
            default=st.session_state.custom_targets
        )
        
        if st.button("Update Targets"):
            st.session_state.custom_targets = all_targets
            st.experimental_rerun()
        
        # Display the custom graph
        st.subheader("Current Custom Graph")
        display_graph(st.session_state.custom_graph, targets=st.session_state.custom_targets)
        
        # Game parameters
        st.subheader("Game Parameters")
        
        lambda_param = st.slider(
            "Wealth Redistribution Factor (λ)",
            min_value=0.0,
            max_value=0.49,
            value=0.125,
            step=0.025
        )
        
        initial_vertex = st.selectbox(
            "Initial Vertex",
            st.session_state.custom_nodes,
            index=0 if st.session_state.custom_nodes else 0
        )
        
        initial_budget = st.slider(
            "Player 1 Initial Budget",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.01
        )
        
        # Run simulation
        if st.button("Run Simulation on Custom Graph"):
            if not st.session_state.custom_targets:
                st.error("Please select at least one target node")
            elif not st.session_state.custom_edges:
                st.error("Please add at least one edge to the graph")
            else:
                with st.spinner("Running simulation..."):
                    game = run_game_simulation(
                        st.session_state.custom_graph, lambda_param, 
                        st.session_state.custom_targets, initial_vertex, initial_budget,
                        optimal_player1_strategy, optimal_player2_strategy
                    )
                
                # Display result
                if game.winner == 1:
                    st.success("Player 1 wins by reaching target!")
                else:
                    st.error("Player 2 wins by preventing target!")
                    
                # Display game history
                st.subheader("Game History")
                display_game_history(game)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About**: This simulator is based on the paper "Robin Hood Reachability Bidding Games" by Shaull Almagor, Guy Avni, and Neta Dafni.
    """)

if __name__ == "__main__":
    main()
