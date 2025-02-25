import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pulp

st.set_page_config(
    page_title="Robin Hood Bidding Games",
    page_icon="🏹",
    layout="wide"
)

st.title("Robin Hood Bidding Game Simulator")

# Apply custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 4px 4px 0px 0px;
        border: 1px solid #dee2e6;
        border-bottom: none;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e7f0ff;
        border-bottom: 2px solid #4361ee;
    }
</style>
""", unsafe_allow_html=True)

# Define the wealth redistribution function
def apply_redistribution(p1_budget, lambda_val):
    """Apply wealth redistribution and return Player 1's new budget"""
    if p1_budget > 0.5:
        # Player 1 is richer
        diff = p1_budget - (1 - p1_budget)
        return p1_budget - lambda_val * diff
    elif p1_budget < 0.5:
        # Player 2 is richer
        diff = (1 - p1_budget) - p1_budget
        return p1_budget + lambda_val * diff
    else:
        # Equal wealth
        return p1_budget

# Create tabs
tab1, tab2, tab3 = st.tabs(["Game Simulation", "Configuration", "History"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Game Graph")
        
        # Create graph
        G = nx.DiGraph()
        G.add_nodes_from(['v_left', 'v_right', 'v_1', 'v_2'])
        G.add_edges_from([
            ('v_left', 'v_right'), ('v_right', 'v_left'),
            ('v_left', 'v_1'), ('v_left', 'v_2')
        ])
        
        # Calculate positions
        pos = {
            'v_left': (-1, 0),
            'v_right': (1, 0),
            'v_1': (-2, -1.5),
            'v_2': (0, -1.5)
        }
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Get current vertex and target from session state
        current_vertex = st.session_state.get('current_vertex', 'v_left')
        target_vertices = st.session_state.get('target_vertices', ['v_1'])
        thresholds = st.session_state.get('thresholds', {
            'v_left': 0.7, 'v_right': 0.5, 'v_1': 0, 'v_2': 1
        })
        
        # Node colors based on status
        node_colors = []
        for node in G.nodes():
            if node == current_vertex:
                node_colors.append('skyblue')
            elif node in target_vertices:
                node_colors.append('lightgreen')
            else:
                node_colors.append('lightgray')
        
        # Draw the graph
        nx.draw(G, pos, with_labels=True, node_color=node_colors, 
                node_size=2000, font_size=12, font_weight='bold',
                arrowsize=20, ax=ax)
        
        # Add threshold labels
        for node, (x, y) in pos.items():
            threshold = thresholds.get(node, 0.5)
            plt.text(x, y-0.3, f"Th: {threshold:.2f}", 
                    fontsize=10, ha='center')
        
        st.pyplot(fig)
        
        # Analysis text
        st.subheader("Analysis")
        lambda_val = st.session_state.get('redistribution_factor', 0.125)
        st.write(f"""
        The visualization above shows the Robin Hood bidding game from the paper. Player 1 aims to reach vertex v_1, 
        while Player 2 aims to prevent this.
        
        At each step, wealth redistribution occurs (with factor λ = {lambda_val:.3f}), where the richer player 
        gives the poorer player a fraction of their wealth advantage.
        
        The threshold for v_left is {thresholds.get('v_left', 0.7):.3f}, meaning Player 1 needs a budget greater 
        than this to win with optimal play.
        """)
        
    with col2:
        st.subheader("Game Status")
        
        # Initialize session state if not already done
        if 'player1_budget' not in st.session_state:
            st.session_state.player1_budget = 0.6
            st.session_state.player2_budget = 0.4
            st.session_state.current_vertex = 'v_left'
            st.session_state.current_player = 1
            st.session_state.game_over = False
            st.session_state.winner = None
            st.session_state.history = []
            st.session_state.redistribution_factor = 0.125
            st.session_state.target_vertices = ['v_1']
        
        # Display current budgets
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown(f"""
            <div style='background-color:#e7f0ff;padding:10px;border-radius:5px;'>
                <div style='color:#6c757d;font-size:0.9em;'>Player 1 Budget</div>
                <div style='font-size:1.5em;font-weight:600;'>{st.session_state.player1_budget:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_p2:
            st.markdown(f"""
            <div style='background-color:#fff0f0;padding:10px;border-radius:5px;'>
                <div style='color:#6c757d;font-size:0.9em;'>Player 2 Budget</div>
                <div style='font-size:1.5em;font-weight:600;'>{st.session_state.player2_budget:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display current position
        st.write("**Current Position:**", st.session_state.current_vertex)
        st.write("**Current Player:**", f"Player {st.session_state.current_player}")
        st.write("**λ (Redistribution Factor):**", f"{st.session_state.redistribution_factor:.3f}")
        
        # Game over state
        if st.session_state.game_over:
            st.success(f"Game Over! Player {st.session_state.winner} wins!")
        else:
            # Calculate maximum bid after redistribution
            redistributed_budget = apply_redistribution(
                st.session_state.player1_budget, 
                st.session_state.redistribution_factor
            )
            
            max_bid = redistributed_budget if st.session_state.current_player == 1 else 1 - redistributed_budget
            
            # Bid input
            bid_amount = st.number_input(
                "Enter bid amount:",
                min_value=0.0,
                max_value=float(max_bid),
                value=min(0.1, max_bid),
                step=0.01,
                format="%.3f"
            )
            
            st.write(f"Maximum bid: {max_bid:.3f}")
            
            # Bid button
            if st.button("Submit Bid"):
                # Apply wealth redistribution
                redistributed_budget_p1 = apply_redistribution(
                    st.session_state.player1_budget, 
                    st.session_state.redistribution_factor
                )
                
                # Calculate opponent's optimal bid (simplified)
                threshold_diff = 0.1  # Simplified value for demo
                opponent_bid = min(threshold_diff, 
                                  (1 - redistributed_budget_p1) if st.session_state.current_player == 1 
                                  else redistributed_budget_p1)
                
                # Determine bid winner
                player1_bid = bid_amount if st.session_state.current_player == 1 else opponent_bid
                player2_bid = opponent_bid if st.session_state.current_player == 1 else bid_amount
                
                # Player 1 wins ties
                player1_wins = player1_bid >= player2_bid
                
                # Update budgets
                if player1_wins:
                    new_budget_p1 = redistributed_budget_p1 - player1_bid
                else:
                    new_budget_p1 = redistributed_budget_p1 + player2_bid
                
                # Choose next vertex based on winner
                if (player1_wins and st.session_state.current_player == 1) or \
                   (not player1_wins and st.session_state.current_player == 2):
                    # Winner's choice - simplified to choose target if possible
                    if st.session_state.current_vertex == 'v_left':
                        next_vertex = 'v_1' if st.session_state.current_player == 1 else 'v_2'
                    else:
                        next_vertex = 'v_left'
                else:
                    # Opponent's choice - simplified
                    if st.session_state.current_vertex == 'v_left':
                        next_vertex = 'v_right'
                    else:
                        next_vertex = 'v_left'
                
                # Check if game is over
                game_over = next_vertex in st.session_state.target_vertices or next_vertex == 'v_2'
                winner = 1 if next_vertex in st.session_state.target_vertices else 2 if next_vertex == 'v_2' else None
                
                # Record history
                st.session_state.history.append({
                    "vertex": st.session_state.current_vertex,
                    "redistributed_budget_p1": redistributed_budget_p1,
                    "player1_bid": player1_bid,
                    "player2_bid": player2_bid,
                    "bid_winner": 1 if player1_wins else 2,
                    "next_vertex": next_vertex,
                    "new_budget_p1": new_budget_p1
                })
                
                # Update game state
                st.session_state.player1_budget = new_budget_p1
                st.session_state.player2_budget = 1 - new_budget_p1
                st.session_state.current_vertex = next_vertex
                st.session_state.current_player = 3 - st.session_state.current_player  # Switch players
                st.session_state.game_over = game_over
                st.session_state.winner = winner
                
                # Rerun to refresh the display
                st.experimental_rerun()
        
        # Game controls
        st.write("---")
        col_reset, col_auto, col_step = st.columns(3)
        
        with col_reset:
            if st.button("🔄 Reset"):
                st.session_state.player1_budget = 0.6
                st.session_state.player2_budget = 0.4
                st.session_state.current_vertex = 'v_left'
                st.session_state.current_player = 1
                st.session_state.game_over = False
                st.session_state.winner = None
                st.session_state.history = []
                st.experimental_rerun()

with tab2:
    st.subheader("Game Configuration")
    
    # Initial budget slider
    initial_budget = st.slider(
        "Initial Budget for Player 1",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.get('initial_budget_p1', 0.6),
        step=0.01,
        format="%.2f"
    )
    
    # Redistribution factor slider
    redistribution_factor = st.slider(
        "Redistribution Factor (λ)",
        min_value=0.0,
        max_value=0.49,
        value=st.session_state.get('redistribution_factor', 0.125),
        step=0.01,
        format="%.3f"
    )
    
    st.write("Note: λ must be less than 0.5 (as per the paper's constraints)")
    
    # Apply button
    if st.button("Apply Settings & Reset Game"):
        st.session_state.player1_budget = initial_budget
        st.session_state.player2_budget = 1 - initial_budget
        st.session_state.redistribution_factor = redistribution_factor
        st.session_state.current_vertex = 'v_left'
        st.session_state.current_player = 1
        st.session_state.game_over = False
        st.session_state.winner = None
        st.session_state.history = []
        
        # Update thresholds based on lambda (simplified calculation)
        st.session_state.thresholds = {
            'v_left': max(0.5, 0.7 - redistribution_factor * 0.4),
            'v_right': min(0.7, 0.5 + redistribution_factor * 0.4),
            'v_1': 0,
            'v_2': 1
        }
        st.experimental_rerun()
    
    st.subheader("Game Explanation")
    st.write("""
    This simulator implements the Robin Hood bidding game described in Figure 1 of the paper. 
    The game is played on a graph with four vertices: v_left, v_right, v_1, and v_2.
    
    **Game Rules:**
    - Player 1 aims to reach v_1, while Player 2 aims to prevent this (either by reaching v_2 or keeping the play in v_left and v_right indefinitely)
    - At each step, before bidding, wealth redistribution occurs: the richer player gives the poorer player a fraction λ of their wealth advantage
    - Players then bid from their budgets, with the higher bidder paying their bid to the opponent and moving the token
    - In case of a tie, Player 1 wins the bidding
    
    **Key Insights:**
    - The game has a threshold value for each vertex - if Player 1's budget exceeds this threshold, they can win; otherwise, Player 2 can win
    - The threshold for v_left varies with the redistribution factor λ, showing discontinuity at λ = 0.25
    - The optimal bidding strategy involves bidding exactly half the difference between the maximum and minimum threshold values among neighboring vertices
    """)

with tab3:
    st.subheader("Game History")
    
    if not st.session_state.get('history', []):
        st.write("No moves yet. Start playing to see the history.")
    else:
        # Create a DataFrame from history for display
        history_data = []
        for i, step in enumerate(st.session_state.history):
            history_data.append({
                "Step": i + 1,
                "Vertex": step["vertex"],
                "After Redistribution": f"{step['redistributed_budget_p1']:.3f}",
                "P1 Bid": f"{step['player1_bid']:.3f}",
                "P2 Bid": f"{step['player2_bid']:.3f}",
                "Winner": f"Player {step['bid_winner']}",
                "Next Vertex": step["next_vertex"],
                "New P1 Budget": f"{step['new_budget_p1']:.3f}"
            })
        
        st.table(history_data)
