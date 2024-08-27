import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from red_black_tree import RedBlackTree

# Load the data
data = pd.read_excel('C:/Users/nmegh/Downloads/proj/testdata.xlsx')

# Convert categorical columns to numerical format
data['Season'] = data['Season'].map({'Winter': 0, 'Summer': 1})
data['Event'] = data['Event'].map({'Yes': 1, 'No': 0})
data['Day-of-week'] = data['Day-of-week'].map({'Weekday': 0, 'Weekend': 1})

# Normalize Historical-occupancy to percentage values if necessary
data['Historical-occupancy'] = data['Historical-occupancy'] / 100

# Define the Bayesian Network structure
model = BayesianNetwork([('Season', 'Historical-occupancy'),
                         ('Event', 'Historical-occupancy'),
                         ('Day-of-week', 'Historical-occupancy')])

# Fit the model using MaximumLikelihoodEstimator
model.fit(data, estimator=MaximumLikelihoodEstimator)

# Perform inference
inference = VariableElimination(model)

def predict_passenger_percentage(season, event, day_of_week):
    # Query the Bayesian Network for the given evidence
    query_result = inference.query(variables=['Historical-occupancy'],
                                   evidence={'Season': season,
                                             'Event': event,
                                             'Day-of-week': day_of_week})
    return query_result

# Streamlit UI
st.title("Passenger Percentage Estimation & Seat Allocation")

# Input handling with mapping
season_input = st.selectbox("Select Season", ["Winter", "Summer"])
event_input = st.selectbox("Is there an Event?", ["Yes", "No"])
day_of_week_input = st.selectbox("Select Day of the Week", ["Weekday", "Weekend"])

# Map string inputs to integers
season = {'Winter': 0, 'Summer': 1}[season_input]
event = {'Yes': 1, 'No': 0}[event_input]
day_of_week = {'Weekday': 0, 'Weekend': 1}[day_of_week_input]

# Predict passenger percentages
probabilities = predict_passenger_percentage(season, event, day_of_week)

# Extract the probabilities
values = probabilities.values
variables = probabilities.variables

# Get the names of the categories from the Bayesian Network
variable_name = variables[0]  # Assuming there is only one variable
outcomes = model.get_cpds(variable_name).state_names[variable_name]

# Calculate the weighted average of the probabilities
estimated_percentage = sum(val * outcome for val, outcome in zip(values, outcomes))

# Display estimated percentage
st.write(f"Estimated Probability of Occupancy: {estimated_percentage * 100:.2f}%")

# Seat allocation
available_seats = st.number_input("Enter the number of available seats", min_value=1, step=1)
required_seats = int(estimated_percentage * available_seats)

rb_tree = RedBlackTree()
for i in range(available_seats):
    rb_tree.insert(i + 1)  # Assuming seats are numbered from 1 to available_seats

allocated_seats = []
for _ in range(required_seats):
    seat = rb_tree.search(min(rb_tree.root.key, rb_tree.root.key))
    if seat:
        allocated_seats.append(seat.key)
        rb_tree.delete_node(seat.key)  # Remove seat from tree

# Display seat allocation results
st.write(f"Number of seats allocated: {len(allocated_seats)}")
st.write(f"Number of seats left: {available_seats - len(allocated_seats)}")

# Draw Bayesian Network structure
def draw_bayesian_network(model):
    G = nx.DiGraph()
    for edge in model.edges():
        G.add_edge(*edge)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue', font_size=8, font_weight='bold', arrows=True)
    plt.title("Bayesian Network Structure")
    st.pyplot(plt)

# Draw the Bayesian Network
st.subheader("Bayesian Network Structure")
draw_bayesian_network(model)

# Draw the Red-Black Tree
def draw_red_black_tree(rbt):
    G = nx.DiGraph()
    pos = {}

    def add_edges(node, x=0, y=0, layer=1):
        if node != rbt.TNULL:
            G.add_node(node.key, pos=(x, y), color=node.color)
            pos[node.key] = (x, y)
            if node.left != rbt.TNULL:
                G.add_edge(node.key, node.left.key)
                add_edges(node.left, x - 1 / layer, y - 1, layer + 1)
            if node.right != rbt.TNULL:
                G.add_edge(node.key, node.right.key)
                add_edges(node.right, x + 1 / layer, y - 1, layer + 1)

    add_edges(rbt.root)

    # Extract node colors
    colors = [nx.get_node_attributes(G, 'color')[node] for node in G.nodes()]
    colors = ['red' if color == 'red' else 'black' for color in colors]

    # Draw the graph
    plt.figure(figsize=(5, 5))
    nx.draw(G, pos=pos, with_labels=True, node_color=colors, node_size=300, font_color='white', font_weight='bold', arrows=False)
    plt.title("Red-Black Tree")
    st.pyplot(plt)

st.subheader("Red-Black Tree Visualization")
draw_red_black_tree(rb_tree)

# Visualize the input and result as nodes
def draw_input_and_result(season, event, day_of_week, estimated_percentage):
    G = nx.DiGraph()

    # Add nodes
    G.add_node("Inputs")
    G.add_node(f"Season: {season_input} ({season})")
    G.add_node(f"Event: {event_input} ({event})")
    G.add_node(f"Day-of-week: {day_of_week_input} ({day_of_week})")
    G.add_node(f"Estimated Percentage: {estimated_percentage * 100:.2f}%")

    # Add edges
    G.add_edges_from([
        ("Inputs", f"Season: {season_input} ({season})"),
        ("Inputs", f"Event: {event_input} ({event})"),
        ("Inputs", f"Day-of-week:  {day_of_week_input}  ({day_of_week})"),
        (f"Season: {season_input} ({season})", f"Estimated Percentage: {estimated_percentage * 100:.2f}%"),
        (f"Event: {event_input} ({event})", f"Estimated Percentage: {estimated_percentage * 100:.2f}%"),
        (f"Day-of-week: {day_of_week_input} ({day_of_week})", f"Estimated Percentage: {estimated_percentage * 100:.2f}%")
    ])

    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=900, node_color='lightgreen', font_size=8, font_weight='bold', arrows=True)
    plt.title("Inputs and Estimated Percentage")
    st.pyplot(plt)

# Draw the input and result
st.subheader("Inputs and Estimated Percentage")
draw_input_and_result(season, event, day_of_week, estimated_percentage)
def draw_red_black_tree(rbt):
    G = nx.DiGraph()
    pos = {}

    def add_edges(node, x=0, y=0, layer=1):
        if node != rbt.TNULL:
            G.add_node(node.key, pos=(x, y), color=node.color)
            pos[node.key] = (x, y)
            if node.left != rbt.TNULL:
                G.add_edge(node.key, node.left.key)
                add_edges(node.left, x - 1 / layer, y - 1, layer + 1)
            if node.right != rbt.TNULL:
                G.add_edge(node.key, node.right.key)
                add_edges(node.right, x + 1 / layer, y - 1, layer + 1)

    add_edges(rbt.root)

    # Extract node colors
    colors = [nx.get_node_attributes(G, 'color')[node] for node in G.nodes()]
    colors = ['red' if color == 'red' else 'black' for color in colors]

    # Draw the graph
    plt.figure(figsize=(5, 5))
    nx.draw(G, pos=pos, with_labels=True, node_color=colors, node_size=300, font_color='white', font_weight='bold', arrows=False)
    plt.title("Red-Black Tree")
    st.pyplot(plt)
