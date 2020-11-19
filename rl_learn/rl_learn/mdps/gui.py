import tkinter as tk
from statistics import median
import numpy as np


def draw_initial(window):
    window.title("Input for MDP")
    window.geometry('500x500')
    
def draw_S_selection():
    S_selection_frame = tk.Frame(master=main_frame, pady=10)
    S_selection_frame.pack()
    S_lbl = tk.Label(S_selection_frame, text="How many states?")
    S_lbl.grid(row=0, column=0)
    S_entry = tk.Entry(S_selection_frame)
    S_entry.grid(row=0, column=1)
    S_button = tk.Button(S_selection_frame, text="Set Number of States", command= lambda: get_S(S_entry))
    S_button.grid(row=0, column=2)

def get_S(S_entry):
    S = S_entry.get()
    try:
        int(S)
        print("You inputted " + S)
    except ValueError:
        print("You provided " + S + " which isn't an integer.")
        return
    
    inputs["S"] = int(S)
    print(inputs)
    draw_A_selection()
    
def draw_A_selection():
    A_selection_frame = tk.Frame(master=main_frame, pady=10)
    A_selection_frame.pack()
    A_lbl = tk.Label(A_selection_frame, text="How many actions?")
    A_lbl.grid(row=0, column=0)
    A_entry = tk.Entry(A_selection_frame)
    A_entry.grid(row=0, column=1)
    A_button = tk.Button(A_selection_frame, text="Set Number of Actions", command= lambda: get_A(A_entry))
    A_button.grid(row=0, column=2)

def get_A(A_entry):
    A = A_entry.get()
    try:
        int(A)
        print("You inputted " + A)
    except ValueError:
        print("You provided " + A + " which isn't an integer.")
        return

    inputs["A"] = int(A)
    print(inputs)
    draw_state_transitions_selection()

def draw_state_transitions_selection():
    transitions_block_frame = tk.Frame(master=main_frame)
    transitions_block_frame.pack()
    transitions_lbl = tk.Label(transitions_block_frame, text="We now require the state transitions probability distribution, p(S_{t+1} = s | A_t = a, S_t = s):")
    transitions_lbl.pack()

    state_transitions_frame = tk.Frame(master=transitions_block_frame, borderwidth=3)
    state_transitions_frame.pack()
    S = inputs["S"]
    A = inputs["A"]
    transition_entries = np.zeros(shape=(S, A, S), dtype=object)

    for s in range(S):
        state_frame = tk.Frame(master=state_transitions_frame, borderwidth=2)
        state_frame.pack()
        
        current_state_label = tk.Label(master=state_frame, text="State Transitions for the Current State " + str(s)  + ":")
        current_state_label.pack()
        
        next_state_label = tk.Label(master=state_frame, text="Next State")
        next_state_label.pack()
        
        matrix_area_frame = tk.Frame(master=state_frame)
        matrix_area_frame.pack(side=tk.LEFT)
        
        action_label = tk.Label(master=matrix_area_frame, text="Action")
        action_label.grid(row=0, column=0)

        actual_matrix = tk.Frame(master=matrix_area_frame)
        actual_matrix.grid(row=0, column=1)

        for i in range(A+1):
            for j in range(S+1):
                if (i == 0 and j == 0):
                    continue
                elif (i == 0 or j == 0):
                    label = tk.Label(master=actual_matrix, text=str(max(i, j)-1))
                    label.grid(row=i, column=j)
                else:
                    elem_frame = tk.Frame(master=actual_matrix, relief=tk.RAISED, borderwidth=1)
                    elem_frame.grid(row=i, column=j)
                    default_value = tk.StringVar(elem_frame, value='0')
                    entry = tk.Entry(elem_frame, textvariable=default_value)
                    transition_entries[s,i-1,j-1] = entry
                    entry.pack()
    
    submit_btn = tk.Button(master=state_transitions_frame, text="Set Transition Probabilities", command= lambda: get_transitions(transition_entries))
    submit_btn.pack()

def get_transitions(transition_entries):
    S = inputs['S']
    A = inputs['A']
    get = np.vectorize(lambda entry: entry.get())
    transitions = get(transition_entries).astype(np.float)
    inputs["Transitions"] = transitions
    print(transitions)
    draw_rewards_selection()

def draw_rewards_selection():
    S = inputs['S']
    A = inputs['A']
    rewards_frame = tk.Frame(master=main_frame)
    rewards_frame.pack()
    rewards_lbl = tk.Label(master=rewards_frame, text="Possible reward values, comma-separated:")
    rewards_lbl.grid(row=0, column=0)
    rewards_entry = tk.Entry(master=rewards_frame)
    rewards_entry.grid(row=0, column=1)
    rewards_button = tk.Button(master=rewards_frame, text="Set rewards", command= lambda: get_rewards(rewards_entry))
    rewards_button.grid(row=0, column=2)

def get_rewards(rewards_entry):
    rewards = [float(reward) for reward in rewards_entry.get().split(',')]
    rewards = sorted(rewards)
    inputs["Rewards"] = rewards
    inputs["R"] = len(rewards)
    print(rewards)

window = tk.Tk()
draw_initial(window)

main_frame = tk.Frame(window, width=100, padx=5, bd=1, relief=tk.RIDGE)
main_frame.pack(pady=(10, 10))
top_lbl = tk.Label(main_frame, text="Welcome! Please specify your MDP using this form.")
top_lbl.pack()
draw_S_selection()
inputs = {}
window.mainloop()